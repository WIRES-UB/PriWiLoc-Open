"""Audit records for AP trust and server-side weight changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from models.secure_agg.types import AttackRecord


@dataclass(frozen=True)
class APTrustEvent:
    """Record one AP's decision about one server response.

    Args:
        ap_id: Access-point identifier.
        epoch: Training epoch index.
        round: Secure aggregation round index.
        step: Global training step.
        trusted_weights: Binary verification decision.
        correct: Cumulative accepted response count.
        incorrect: Cumulative rejected response count.
        trust_score: Cumulative accepted-response fraction.
    """

    ap_id: int
    epoch: int
    round: int
    step: int
    trusted_weights: int
    correct: int
    incorrect: int
    trust_score: float


class APTrustTracker:
    """Track bounded counters and pending trust events for one AP.

    Args:
        ap_id: Access-point identifier owned by the tracker.
    """

    def __init__(self, ap_id: int):
        """Initialize empty trust counters and pending history.

        Args:
            ap_id: Access-point identifier owned by the tracker.

        Returns:
            None.
        """
        self.ap_id = int(ap_id)
        self.correct = 0
        self.incorrect = 0
        self._history: list[APTrustEvent] = []

    @property
    def history(self) -> tuple[APTrustEvent, ...]:
        """Return an immutable snapshot of pending audit events.

        Returns:
            Pending AP trust events.
        """

        return tuple(self._history)

    @property
    def trust_score(self) -> float:
        """Calculate the accepted-response fraction.

        Returns:
            The trust score, or zero before any response.
        """
        total = self.correct + self.incorrect
        return self.correct / total if total else 0.0

    def record(
        self,
        *,
        epoch: int,
        round_idx: int,
        trusted_weights: bool,
        step_idx: int | None = None,
    ) -> APTrustEvent:
        """Record one verification decision and update cumulative counters.

        Args:
            epoch: Training epoch index.
            round_idx: Secure aggregation round index.
            trusted_weights: Whether the AP accepted the server response.
            step_idx: Optional global training step.

        Returns:
            The newly recorded trust event.
        """
        if trusted_weights:
            self.correct += 1
        else:
            self.incorrect += 1
        event = APTrustEvent(
            ap_id=self.ap_id,
            epoch=int(epoch),
            round=int(round_idx),
            step=int(round_idx if step_idx is None else step_idx),
            trusted_weights=int(trusted_weights),
            correct=self.correct,
            incorrect=self.incorrect,
            trust_score=self.trust_score,
        )
        self._history.append(event)
        return event

    def state_dict(self) -> dict[str, Any]:
        """Serialize bounded state required to resume trust accounting.

        Returns:
            AP identity and cumulative trust counters.
        """

        return {
            "ap_id": self.ap_id,
            "correct": self.correct,
            "incorrect": self.incorrect,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore bounded counters without restoring event history.

        Args:
            state: Previously serialized tracker state.

        Returns:
            None.
        """

        if int(state["ap_id"]) != self.ap_id:
            raise ValueError(
                f"Cannot load AP {state['ap_id']} trust state into AP {self.ap_id}."
            )
        self.correct = int(state["correct"])
        self.incorrect = int(state["incorrect"])
        self._history.clear()

    def drain_history(self, count: int | None = None) -> tuple[APTrustEvent, ...]:
        """Remove and return successfully exported pending events.

        Args:
            count: Number of oldest events to remove, or all when omitted.

        Returns:
            The removed trust events.
        """

        resolved = len(self._history) if count is None else int(count)
        if resolved < 0 or resolved > len(self._history):
            raise ValueError("Drain count is outside the pending-event range.")
        drained = tuple(self._history[:resolved])
        del self._history[:resolved]
        return drained


@dataclass(frozen=True)
class ServerWeightChangeEvent:
    """Record AP responses intentionally changed in one server round.

    Args:
        epoch: Training epoch index.
        round: Secure aggregation round index.
        step: Global training step.
        changed_ap_ids: AP responses changed effectively.
        attack_records: Attack rules executed in the round.
    """

    epoch: int
    round: int
    step: int
    changed_ap_ids: tuple[int, ...]
    attack_records: tuple[AttackRecord, ...] = ()


class ServerWeightChangeTracker:
    """Track pending server audit events, including unchanged rounds."""

    def __init__(self):
        """Initialize an empty pending server-event history.

        Returns:
            None.
        """
        self._history: list[ServerWeightChangeEvent] = []

    @property
    def history(self) -> tuple[ServerWeightChangeEvent, ...]:
        """Return an immutable snapshot of pending server audit events.

        Returns:
            Pending server response-change events.
        """

        return tuple(self._history)

    def record(
        self,
        *,
        epoch: int,
        round_idx: int,
        changed_ap_ids: Iterable[int],
        step_idx: int | None = None,
        attack_records: Iterable[AttackRecord] = (),
    ) -> ServerWeightChangeEvent:
        """Record one round of server response changes.

        Args:
            epoch: Training epoch index.
            round_idx: Secure aggregation round index.
            changed_ap_ids: AP responses changed effectively.
            step_idx: Optional global training step.
            attack_records: Attack rules executed in the round.

        Returns:
            The newly recorded server event.
        """
        event = ServerWeightChangeEvent(
            epoch=int(epoch),
            round=int(round_idx),
            step=int(round_idx if step_idx is None else step_idx),
            changed_ap_ids=tuple(sorted({int(ap_id) for ap_id in changed_ap_ids})),
            attack_records=tuple(attack_records),
        )
        self._history.append(event)
        return event

    def drain_history(
        self,
        count: int | None = None,
    ) -> tuple[ServerWeightChangeEvent, ...]:
        """Remove and return successfully exported pending events.

        Args:
            count: Number of oldest events to remove, or all when omitted.

        Returns:
            The removed server events.
        """

        resolved = len(self._history) if count is None else int(count)
        if resolved < 0 or resolved > len(self._history):
            raise ValueError("Drain count is outside the pending-event range.")
        drained = tuple(self._history[:resolved])
        del self._history[:resolved]
        return drained
