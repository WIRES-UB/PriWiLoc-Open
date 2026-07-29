"""Evaluation statistics, CDF plots, and reproducible offline assets."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

TORCH_QUANTILE_MAX_ELEMENTS = 16_700_000


def safe_percentiles(
    errors: torch.Tensor,
    percentiles: Iterable[float],
) -> dict[float, float]:
    """Compute exact percentiles with a NumPy fallback for large tensors.

    Args:
        errors: Tensor containing scalar errors.
        percentiles: Percentile values in the closed interval [0, 100].

    Returns:
        A mapping from requested percentiles to computed values.
    """
    flat = errors.detach().reshape(-1).cpu()
    requested = [float(value) for value in percentiles]
    if flat.numel() == 0:
        raise ValueError("Cannot compute percentiles for an empty error tensor.")
    if any(value < 0 or value > 100 for value in requested):
        raise ValueError("Percentiles must be in the closed interval [0, 100].")

    if flat.numel() <= TORCH_QUANTILE_MAX_ELEMENTS:
        quantiles = torch.tensor(
            [value / 100.0 for value in requested],
            dtype=flat.dtype,
        )
        values = torch.quantile(flat, quantiles).tolist()
    else:
        values = np.percentile(flat.numpy(), requested).tolist()

    return dict(zip(requested, (float(value) for value in values)))


def create_cdf_figure(
    errors_by_dataset: Mapping[str, np.ndarray],
    *,
    max_points: Optional[int] = 20_000,
    title: str = "Location error CDF",
):
    """Create a deterministic empirical CDF without subsampling statistics.

    Args:
        errors_by_dataset: Dataset names mapped to raw error arrays.
        max_points: Maximum plotted points per dataset.
        title: Figure title.

    Returns:
        A Matplotlib CDF figure.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, raw_errors in errors_by_dataset.items():
        errors = np.asarray(raw_errors, dtype=np.float64).reshape(-1)
        if errors.size == 0:
            continue
        sorted_errors = np.sort(errors)
        full_indices = np.arange(sorted_errors.size)
        if max_points and sorted_errors.size > max_points:
            selected = np.linspace(
                0,
                sorted_errors.size - 1,
                num=max_points,
                dtype=np.int64,
            )
            plot_errors = sorted_errors[selected]
            cdf = (full_indices[selected] + 1) / sorted_errors.size
        else:
            plot_errors = sorted_errors
            cdf = (full_indices + 1) / sorted_errors.size
        axis.plot(plot_errors, cdf, label=name)

    axis.set_xlabel("Location error")
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if errors_by_dataset:
        axis.legend()
    figure.tight_layout()
    return figure


def sanitize_name(value: str) -> str:
    """Convert a value to a stable filename-safe name.

    Args:
        value: Value to sanitize.

    Returns:
        A non-empty filename-safe string.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def save_dataset_assets(
    *,
    output_dir: str,
    run_name: str,
    dataset_name: str,
    errors: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    raw_error_format: str,
) -> list[Path]:
    """Persist raw errors, predictions, and targets for offline analysis.

    Args:
        output_dir: Destination directory.
        run_name: Timestamped experiment name.
        dataset_name: Evaluation dataset name.
        errors: Per-sample location errors.
        predictions: Model location predictions.
        targets: Ground-truth locations.
        raw_error_format: Error format: `npy`, `csv`, or `both`.

    Returns:
        Paths of all written artifact files.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{sanitize_name(run_name)}_{sanitize_name(dataset_name)}"
    errors_np = errors.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    file_format = raw_error_format.lower()
    if file_format not in {"npy", "csv", "both"}:
        raise ValueError("eval.raw_error_format must be one of: npy, csv, both.")

    paths: list[Path] = []
    if file_format in {"npy", "both"}:
        error_path = destination / f"errors_{stem}.npy"
        np.save(error_path, errors_np)
        paths.append(error_path)
    if file_format in {"csv", "both"}:
        error_path = destination / f"errors_{stem}.csv"
        np.savetxt(error_path, errors_np.reshape(-1, 1), delimiter=",", header="error", comments="")
        paths.append(error_path)

    predictions_path = destination / f"predictions_{stem}.npy"
    targets_path = destination / f"targets_{stem}.npy"
    np.save(predictions_path, predictions_np)
    np.save(targets_path, targets_np)
    paths.extend([predictions_path, targets_path])
    return paths


def save_summary_csv(
    *,
    output_dir: str,
    run_name: str,
    rows: list[dict[str, object]],
) -> Path:
    """Write per-dataset summary metrics to a CSV asset.

    Args:
        output_dir: Destination directory.
        run_name: Timestamped experiment name.
        rows: Summary rows keyed by metric name.

    Returns:
        Path to the written CSV file.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"summary_metrics_{sanitize_name(run_name)}.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
