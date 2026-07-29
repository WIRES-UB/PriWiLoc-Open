"""Matplotlib figures for immutable secure-aggregation audit records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from models.secure_agg.trust_tracking import (
    APTrustEvent,
    ServerWeightChangeEvent,
)


def _format_binary_axis(axis, *, ylabel: str) -> None:
    """Apply shared formatting to a binary event axis.

    Args:
        axis: Matplotlib axis to format.
        ylabel: Label for the vertical axis.

    Returns:
        None.
    """

    axis.set_ylim(-0.1, 1.1)
    axis.set_yticks([0, 1])
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="x", alpha=0.25)


def create_ap_binary_trust_figure(
    history: Mapping[int, Sequence[APTrustEvent]],
) -> Figure:
    """Create a figure of per-AP binary trust decisions.

    Args:
        history: AP identifiers mapped to trust-event histories.

    Returns:
        A Matplotlib figure containing binary trust decisions.
    """

    figure, axis = plt.subplots(figsize=(12, 4.5))
    for ap_id, events in sorted(history.items()):
        axis.step(
            [event.step for event in events],
            [event.trusted_weights for event in events],
            where="post",
            marker="o",
            markersize=3,
            label=f"AP {ap_id}",
        )
    _format_binary_axis(axis, ylabel="Trust decision")
    axis.set_xlabel("Training step")
    axis.set_title("AP trust in server weights by step")
    axis.legend(loc="best", ncol=max(1, min(4, len(history))))
    figure.tight_layout()
    return figure


def create_ap_cumulative_trust_figure(
    history: Mapping[int, Sequence[APTrustEvent]],
) -> Figure:
    """Create a figure of cumulative per-AP trust scores.

    Args:
        history: AP identifiers mapped to trust-event histories.

    Returns:
        A Matplotlib figure containing cumulative trust scores.
    """

    figure, axis = plt.subplots(figsize=(12, 4.5))
    for ap_id, events in sorted(history.items()):
        axis.plot(
            [event.step for event in events],
            [event.trust_score for event in events],
            marker="o",
            markersize=3,
            label=f"AP {ap_id}",
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cumulative trust score")
    axis.set_title("AP cumulative trust by step")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", ncol=max(1, min(4, len(history))))
    figure.tight_layout()
    return figure


def create_server_attack_figure(
    history: Sequence[ServerWeightChangeEvent],
    *,
    n_aps: int,
) -> Figure:
    """Create vertically stacked server-attack timelines for every AP.

    Args:
        history: Ordered server response-change events.
        n_aps: Number of access points represented in the figure.

    Returns:
        A Matplotlib figure containing AP-specific attack timelines.
    """

    if n_aps <= 0:
        raise ValueError("n_aps must be positive.")
    figure, axes = plt.subplots(
        n_aps,
        1,
        sharex=True,
        figsize=(12, 2 * n_aps + 1),
        squeeze=False,
    )
    steps = [event.step for event in history]
    for ap_id, axis in enumerate(axes[:, 0]):
        axis.step(
            steps,
            [int(ap_id in event.changed_ap_ids) for event in history],
            where="post",
            marker="o",
            markersize=3,
            color=f"C{ap_id % 10}",
        )
        _format_binary_axis(axis, ylabel=f"AP {ap_id}")
        axis.set_title(
            f"Server attack targeting AP {ap_id}",
            loc="left",
            fontsize=10,
        )
    axes[-1, 0].set_xlabel("Training step")
    figure.suptitle("Server attacks by AP and training step")
    figure.tight_layout()
    return figure
