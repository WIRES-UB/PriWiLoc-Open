"""Federated-learning model for the secure-aggregation experiment pipeline."""

from __future__ import annotations

import torch

from models.federated_learning import FederatedLearningModel
from models.aggregation import (
    AggregationStrategy,
    ClientUpdate,
    UniformMeanAggregation,
)
from models.secure_agg import APTrustTracker, AggregationReport
from models.secure_agg.factory import build_secure_aggregator
from utils.config_hydra import Config


class SecureFederatedLearningModel(FederatedLearningModel):
    """Add flag-gated secure aggregation to the federated model.

    Args:
        config: Root experiment configuration.
        aggregation_strategy: Optional prebuilt shared aggregation strategy.
    """

    def __init__(
        self,
        config: Config,
        aggregation_strategy: AggregationStrategy | None = None,
    ):
        """Initialize secure protocol state and per-AP trust trackers.

        Args:
            config: Root experiment configuration.
            aggregation_strategy: Optional prebuilt shared aggregation strategy.

        Returns:
            None.
        """
        strategy = aggregation_strategy or UniformMeanAggregation(
            reference_client_id=0,
        )
        super().__init__(config, aggregation_strategy=strategy)
        self.secure_aggregator = build_secure_aggregator(
            config,
            strategy=strategy,
        )
        if self.secure_aggregator is not None:
            self.secure_aggregator.validate_configuration(
                self.resnet_encoder_list[0].state_dict(),
                n_aps=len(self.resnet_encoder_list),
            )
        self.ap_trust_trackers = [
            APTrustTracker(ap_id)
            for ap_id in range(len(self.resnet_encoder_list))
        ]
        self.register_buffer(
            "_secure_round_idx",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_secure_rejections_cumulative",
            torch.tensor(0, dtype=torch.long),
            persistent=True,
        )

    def _log_secure_report(self, report: AggregationReport) -> None:
        """Log round diagnostics and update AP trust trackers.

        Args:
            report: Completed secure aggregation report.

        Returns:
            None.
        """
        metrics = {
            "secure/accepted": float(report.accepted),
            "secure/server_honest": float(report.server_honest),
            "secure/recover_max_abs_err": report.recover_max_abs_err,
            "secure/quant_rel_err": report.quant_rel_err,
            "secure/rejections_cumulative": float(report.rejections_cumulative),
            "secure/round_idx": float(report.round),
            "secure/round_seconds": report.round_seconds,
            "secure/server_changed_ap_count": float(len(report.changed_ap_ids)),
        }
        changed_ap_ids = set(report.changed_ap_ids)
        for ap_id, trusted_weights in enumerate(report.ap_trust):
            tracker = self.ap_trust_trackers[ap_id]
            event = tracker.record(
                epoch=report.epoch,
                round_idx=report.round,
                trusted_weights=trusted_weights,
                step_idx=report.step,
            )
            metrics[f"secure/ap_{ap_id}/trust_binary"] = float(
                event.trusted_weights
            )
            metrics[f"secure/ap_{ap_id}/trust_score"] = event.trust_score
            metrics[f"secure/ap_{ap_id}/correct"] = float(event.correct)
            metrics[f"secure/ap_{ap_id}/incorrect"] = float(event.incorrect)
            metrics[f"secure/server_changed_ap_{ap_id}"] = float(
                ap_id in changed_ap_ids
            )
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)

    @property
    def server_weight_change_history(self):
        """Return an immutable snapshot of pending server audit events.

        Returns:
            Pending server response-change events.
        """

        if self.secure_aggregator is None:
            return ()
        return self.secure_aggregator.weight_change_history

    def get_ap_trust_history(self) -> dict[int, tuple]:
        """Return immutable snapshots of pending per-AP trust events.

        Returns:
            AP identifiers mapped to pending trust events.
        """

        return {
            tracker.ap_id: tracker.history
            for tracker in self.ap_trust_trackers
        }

    def on_train_epoch_end(self) -> None:
        """Log bounded trust counters at the end of each epoch.

        Returns:
            None.
        """

        super().on_train_epoch_end()
        if self.secure_aggregator is None:
            return
        self.log_dict(
            {
                f"secure/ap_{tracker.ap_id}/trust_score_through_epoch": (
                    tracker.trust_score
                )
                for tracker in self.ap_trust_trackers
            },
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    def on_save_checkpoint(self, checkpoint) -> None:
        """Persist bounded protocol and trust resume state.

        Args:
            checkpoint: Mutable Lightning checkpoint payload.

        Returns:
            None.
        """

        checkpoint["secure_resume_state"] = {
            "round_idx": int(self._secure_round_idx.item()),
            "rejections_cumulative": int(
                self._secure_rejections_cumulative.item()
            ),
            "aps": [tracker.state_dict() for tracker in self.ap_trust_trackers],
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        """Restore bounded protocol and trust resume state.

        Args:
            checkpoint: Loaded Lightning checkpoint payload.

        Returns:
            None.
        """

        state = checkpoint.get("secure_resume_state")
        if state is None:
            return
        ap_states = state.get("aps", [])
        if len(ap_states) != len(self.ap_trust_trackers):
            raise ValueError(
                "Checkpoint AP trust state does not match the configured AP count."
        )
        for tracker, tracker_state in zip(self.ap_trust_trackers, ap_states):
            tracker.load_state_dict(tracker_state)
        self._secure_round_idx.fill_(int(state.get("round_idx", 0)))
        self._secure_rejections_cumulative.fill_(
            int(state.get("rejections_cumulative", 0))
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Run plain aggregation or the secure protocol at round boundaries.

        Args:
            outputs: Training-batch output; unused.
            batch: Current training batch.
            batch_idx: Zero-based batch index within the epoch.

        Returns:
            None.
        """
        if (batch_idx + 1) % self.config.model.average_weight_every_n_batches != 0:
            return

        self.print(
            f"epoch: {self.trainer.current_epoch}, batch: {batch_idx}, "
            "averaging client weight"
        )
        client_parameters = self.get_resnet_encoder_parameters()
        weights = self.get_aggregation_weights(
            batch,
            len(client_parameters),
        )
        if self.secure_aggregator is None:
            averaged_parameters = self.aggregate_resnet_encoder_parameters(
                client_parameters,
                weights,
            )
            self.set_resnet_encoder_parameters(averaged_parameters)
            return

        round_idx = int(self._secure_round_idx.item())
        self.secure_aggregator.rejections_cumulative = int(
            self._secure_rejections_cumulative.item()
        )
        updates = [
            ClientUpdate(
                client_id=ap_id,
                state=state,
                weight=weight,
            )
            for ap_id, (state, weight) in enumerate(
                zip(client_parameters, weights)
            )
        ]
        responses, report = self.secure_aggregator.aggregate(
            updates,
            round_idx=round_idx,
            epoch_idx=int(self.trainer.current_epoch),
            step_idx=int(self.trainer.global_step),
        )
        self._log_secure_report(report)
        self._secure_round_idx += 1
        self._secure_rejections_cumulative.fill_(
            report.rejections_cumulative
        )
        for response in responses:
            if response.verified:
                self.set_resnet_encoder_parameters(
                    dict(response.state),
                    [response.ap_id],
                )
        if not any(response.verified for response in responses):
            self.print(
                f"secure aggregation round {round_idx} rejected; "
                "keeping local encoder weights"
            )
