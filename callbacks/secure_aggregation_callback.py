"""Presentation and durable audit export for secure aggregation."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

from models.secure_agg.trust_tracking import (
    APTrustEvent,
    ServerWeightChangeEvent,
)
from models.secure_agg.types import AttackRecord
from utils.experiment_logging import log_asset, log_figure
from utils.secure_agg_plotting import (
    create_ap_binary_trust_figure,
    create_ap_cumulative_trust_figure,
    create_server_attack_figure,
)


class SecureAggregationCallback(Callback):
    """Log trust figures and export immutable audit-event chunks.

    Args:
        output_dir: Directory receiving audit chunks.
        plot_interval_epochs: Epochs between trust figure updates.
        export_interval_epochs: Epochs between audit exports.
        max_pending_events: Event count that triggers early export.
    """

    def __init__(
        self,
        *,
        output_dir: str = "secure_audit",
        plot_interval_epochs: int = 1,
        export_interval_epochs: int = 1,
        max_pending_events: int = 10_000,
    ) -> None:
        """Initialize audit output, plotting intervals, and segment identity.

        Args:
            output_dir: Directory receiving audit chunks.
            plot_interval_epochs: Epochs between trust figure updates.
            export_interval_epochs: Epochs between audit exports.
            max_pending_events: Event count that triggers early export.

        Returns:
            None.
        """
        super().__init__()
        if plot_interval_epochs <= 0 or export_interval_epochs <= 0:
            raise ValueError("Secure callback intervals must be positive.")
        if max_pending_events <= 0:
            raise ValueError("max_pending_events must be positive.")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.plot_interval_epochs = int(plot_interval_epochs)
        self.export_interval_epochs = int(export_interval_epochs)
        self.max_pending_events = int(max_pending_events)
        self.segment_id = uuid.uuid4().hex
        self.parent_segment_id: str | None = None
        self.segment_lineage: tuple[tuple[str, int, str], ...] = ()
        self.export_index = 0

    def state_dict(self) -> dict[str, object]:
        """Serialize bounded segment metadata for checkpoint linkage.

        Returns:
            Current segment identity, lineage, boundary, and artifact root.
        """

        return {
            "segment_id": self.segment_id,
            "segment_lineage": self.segment_lineage,
            "export_index": self.export_index,
            "output_dir": str(self.output_dir),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Start a new audit segment linked to a checkpoint segment.

        Args:
            state_dict: Previously serialized callback segment state.

        Returns:
            None.
        """

        parent = state_dict.get("segment_id")
        self.parent_segment_id = str(parent) if parent else None
        lineage = tuple(
            (
                str(segment),
                int(cutoff),
                str(root),
            )
            for segment, cutoff, root in state_dict.get(
                "segment_lineage",
                (),
            )
        )
        parent_cutoff = int(state_dict.get("export_index", 0))
        parent_root = str(
            state_dict.get("output_dir", self.output_dir)
        )
        self.segment_lineage = (
            (
                *lineage,
                (
                    self.parent_segment_id,
                    parent_cutoff,
                    parent_root,
                ),
            )
            if self.parent_segment_id
            else lineage
        )
        self.segment_id = uuid.uuid4().hex
        self.export_index = 0

    @staticmethod
    def _secure_model(pl_module):
        """Resolve the model only when secure aggregation is enabled.

        Args:
            pl_module: Lightning module supplied to the callback.

        Returns:
            The secure model, or `None` when secure aggregation is disabled.
        """

        return (
            pl_module
            if getattr(pl_module, "secure_aggregator", None) is not None
            else None
        )

    def _log_figures(self, trainer, pl_module) -> None:
        """Create, log, and close secure audit figures.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.

        Returns:
            None.
        """

        model = self._secure_model(pl_module)
        if (
            model is None
            or not getattr(trainer, "is_global_zero", True)
        ):
            return
        trust_history, server_history = self._complete_history(model)
        if not server_history:
            return
        builders = (
            (
                "secure_ap_trust_binary_by_step",
                lambda: create_ap_binary_trust_figure(trust_history),
            ),
            (
                "secure_ap_trust_score_by_step",
                lambda: create_ap_cumulative_trust_figure(trust_history),
            ),
            (
                "secure_server_attacks_by_ap_and_step",
                lambda: create_server_attack_figure(
                    server_history,
                    n_aps=len(model.ap_trust_trackers),
                ),
            ),
        )
        for name, build_figure in builders:
            figure = build_figure()
            try:
                log_figure(
                    trainer.logger,
                    name,
                    figure,
                    step=int(trainer.global_step),
                    overwrite=True,
                )
            finally:
                plt.close(figure)

    @staticmethod
    def _server_event(payload: dict) -> ServerWeightChangeEvent:
        """Deserialize one exported server response-change event.

        Args:
            payload: JSON-compatible event payload.

        Returns:
            The reconstructed immutable server event.
        """

        records = tuple(
            AttackRecord(
                **{
                    **record,
                    "ap_ids": tuple(record["ap_ids"]),
                    "tensor_keys": tuple(record["tensor_keys"]),
                }
            )
            for record in payload.get("attack_records", [])
        )
        return ServerWeightChangeEvent(
            epoch=int(payload["epoch"]),
            round=int(payload["round"]),
            step=int(payload["step"]),
            changed_ap_ids=tuple(payload["changed_ap_ids"]),
            attack_records=records,
        )

    def _complete_history(self, model):
        """Load exported lineage events and append pending runtime events.

        Args:
            model: Secure federated model owning pending trackers.

        Returns:
            Complete AP trust histories and server change history.
        """

        trust: dict[int, list[APTrustEvent]] = {
            tracker.ap_id: [] for tracker in model.ap_trust_trackers
        }
        server: list[ServerWeightChangeEvent] = []
        segments = (
            *self.segment_lineage,
            (
                self.segment_id,
                self.export_index,
                str(self.output_dir),
            ),
        )
        for segment, cutoff, root in segments:
            for path in sorted(
                Path(root).glob(f"audit_{segment}_*.jsonl")
            ):
                chunk_index = int(path.stem.rsplit("_", 1)[-1])
                if chunk_index >= cutoff:
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    payload = record["payload"]
                    if record["event_type"] == "ap_trust":
                        event = APTrustEvent(**payload)
                        trust.setdefault(event.ap_id, []).append(event)
                    elif record["event_type"] == "server_response":
                        server.append(self._server_event(payload))
        for tracker in model.ap_trust_trackers:
            trust.setdefault(tracker.ap_id, []).extend(tracker.history)
        server.extend(model.server_weight_change_history)
        return (
            {
                ap_id: tuple(events)
                for ap_id, events in trust.items()
            },
            tuple(server),
        )

    def _pending_events(self, model) -> tuple[list[dict], list[int], int]:
        """Serialize pending events without mutating their trackers.

        Args:
            model: Secure federated model owning pending trackers.

        Returns:
            Serialized records, per-AP counts, and server-event count.
        """

        records: list[dict] = []
        ap_counts: list[int] = []
        for tracker in model.ap_trust_trackers:
            events = tracker.history
            ap_counts.append(len(events))
            records.extend(
                {
                    "event_type": "ap_trust",
                    "segment_id": self.segment_id,
                    "parent_segment_id": self.parent_segment_id,
                    "payload": asdict(event),
                }
                for event in events
            )
        server_events = model.server_weight_change_history
        records.extend(
            {
                "event_type": "server_response",
                "segment_id": self.segment_id,
                "parent_segment_id": self.parent_segment_id,
                "payload": asdict(event),
            }
            for event in server_events
        )
        return records, ap_counts, len(server_events)

    def _drain_events(
        self,
        model,
        ap_counts: list[int],
        server_count: int,
    ) -> None:
        """Drain exactly the events included in a successful export.

        Args:
            model: Secure federated model owning pending trackers.
            ap_counts: Exported event count for each AP tracker.
            server_count: Exported server-event count.

        Returns:
            None.
        """

        for tracker, count in zip(model.ap_trust_trackers, ap_counts):
            tracker.drain_history(count)
        model.secure_aggregator.weight_change_tracker.drain_history(
            server_count
        )

    def _export_events(self, trainer, pl_module) -> Path | None:
        """Write pending events to a new immutable JSONL chunk.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.

        Returns:
            The exported chunk path, or `None` when nothing was written.
        """

        model = self._secure_model(pl_module)
        if model is None:
            return None
        records, ap_counts, server_count = self._pending_events(model)
        if not records:
            return None
        if not getattr(trainer, "is_global_zero", True):
            self._drain_events(model, ap_counts, server_count)
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"audit_{self.segment_id}_{self.export_index:06d}.jsonl"
        )
        destination = self.output_dir / filename
        if destination.exists():
            raise FileExistsError(
                f"Audit chunk already exists: {destination}."
            )
        temporary = self.output_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                for sequence, record in enumerate(records):
                    record["sequence"] = sequence
                    handle.write(json.dumps(record, sort_keys=True))
                    handle.write("\n")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        self._drain_events(model, ap_counts, server_count)
        self.export_index += 1
        log_asset(
            trainer.logger,
            destination,
            metadata={
                "segment_id": self.segment_id,
                "parent_segment_id": self.parent_segment_id or "",
            },
        )
        return destination

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        """Export early when pending events reach the memory bound.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.
            outputs: Training-batch output; unused.
            batch: Training batch; unused.
            batch_idx: Training batch index; unused.

        Returns:
            None.
        """

        del outputs, batch, batch_idx
        model = self._secure_model(pl_module)
        if model is None:
            return
        pending = sum(
            len(tracker.history) for tracker in model.ap_trust_trackers
        ) + len(model.server_weight_change_history)
        if pending >= self.max_pending_events:
            self._export_events(trainer, model)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        """Log figures and export events at configured epoch intervals.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.

        Returns:
            None.
        """

        epoch_number = int(trainer.current_epoch) + 1
        if epoch_number % self.plot_interval_epochs == 0:
            self._log_figures(trainer, pl_module)
        if epoch_number % self.export_interval_epochs == 0:
            self._export_events(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        """Export pending events before a checkpoint is finalized.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.
            checkpoint: Mutable Lightning checkpoint payload.

        Returns:
            None.
        """

        self._export_events(trainer, pl_module)
        callback_states = checkpoint.get("callbacks")
        if isinstance(callback_states, dict):
            callback_states[self.state_key] = self.state_dict()

    def on_fit_end(self, trainer, pl_module) -> None:
        """Export final events not handled by an epoch interval.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.

        Returns:
            None.
        """

        self._export_events(trainer, pl_module)
