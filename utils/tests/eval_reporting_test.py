import numpy as np
import torch

from utils.eval_reporting import (
    create_cdf_figure,
    safe_percentiles,
    save_dataset_assets,
    save_summary_csv,
)


def test_percentiles_cdf_and_assets(tmp_path):
    errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
    percentiles = safe_percentiles(errors, [50, 75, 90])
    assert percentiles[50.0] == 2.5
    assert percentiles[75.0] == 3.25

    figure = create_cdf_figure({"d1": errors.numpy()}, max_points=2)
    assert figure.axes[0].get_ylabel() == "Empirical CDF"

    paths = save_dataset_assets(
        output_dir=str(tmp_path),
        run_name="r2_secure_honest",
        dataset_name="d1",
        errors=errors,
        predictions=torch.zeros(4, 2),
        targets=torch.ones(4, 2),
        raw_error_format="both",
    )
    assert len(paths) == 4
    assert all(path.exists() for path in paths)
    assert np.load(next(path for path in paths if path.name.startswith("errors_") and path.suffix == ".npy")).shape == (4,)

    summary = save_summary_csv(
        output_dir=str(tmp_path),
        run_name="r2_secure_honest",
        rows=[{"dataset": "d1", "median": 2.5}],
    )
    assert summary.exists()
