"""Shared adapters for experiment figures and file assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)


def log_figure(
    logger,
    figure_name: str,
    figure,
    *,
    step: int | None = None,
    overwrite: bool = False,
) -> None:
    """Log a Matplotlib figure through a supported experiment backend.

    Args:
        logger: Lightning logger or `None`.
        figure_name: Display name for the figure.
        figure: Matplotlib figure to log.
        step: Optional training step associated with the figure.
        overwrite: Whether a same-named remote figure may be replaced.

    Returns:
        None.
    """

    experiment = getattr(logger, "experiment", None) if logger else None
    if experiment is None:
        return
    if hasattr(experiment, "log_figure"):
        kwargs: dict[str, Any] = {
            "figure_name": figure_name,
            "figure": figure,
        }
        if step is not None:
            kwargs["step"] = int(step)
        if overwrite:
            kwargs["overwrite"] = True
        experiment.log_figure(**kwargs)
    elif hasattr(experiment, "add_figure"):
        experiment.add_figure(figure_name, figure, global_step=step)
    elif hasattr(experiment, "log"):
        import wandb

        experiment.log({figure_name: wandb.Image(figure)}, step=step)


def log_figure_and_close(
    logger,
    figure_name: str,
    figure,
    *,
    step: int | None = None,
    overwrite: bool = False,
) -> None:
    """Log a figure and always release its Matplotlib resources.

    Args:
        logger: Lightning logger or `None`.
        figure_name: Display name for the figure.
        figure: Matplotlib figure to log and close.
        step: Optional training step associated with the figure.
        overwrite: Whether a same-named remote figure may be replaced.

    Returns:
        None.
    """

    try:
        log_figure(
            logger,
            figure_name,
            figure,
            step=step,
            overwrite=overwrite,
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def log_asset(
    logger,
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    """Log a file asset through a supported experiment backend.

    Args:
        logger: Lightning logger or `None`.
        path: Local asset path.
        metadata: Optional metadata attached to the asset.
        overwrite: Whether a same-named remote asset may be replaced.

    Returns:
        None.
    """

    experiment = getattr(logger, "experiment", None) if logger else None
    if experiment is None:
        return
    resolved = str(path)
    if hasattr(experiment, "log_asset"):
        kwargs: dict[str, object] = {}
        if metadata is not None:
            kwargs["metadata"] = dict(metadata)
        if overwrite:
            kwargs["overwrite"] = True
        experiment.log_asset(resolved, **kwargs)
    elif hasattr(experiment, "save"):
        experiment.save(resolved)
