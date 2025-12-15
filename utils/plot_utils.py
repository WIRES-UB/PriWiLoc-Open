"""Utility functions for plotting data."""
from typing import Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.schema import AoAVisualizationSample

def plot_ray(
    ray_origins: Union[torch.Tensor, np.ndarray],
    ray_angles: Union[torch.Tensor, np.ndarray],
    length: float = 10,
) -> None:
    """
    Plots rays starting at points specified in ray_origins with angles specified in ray_angles.

    Args:
        ray_origins: Tensor of shape (n, 2) containing the x and y coordinates of the ray starting points.
        ray_angles: Tensor of shape (n,) containing the angles in radians with respect to the horizontal axis.
        length: Length of the rays. Defaults to 10.
    """
    assert (
        ray_origins.shape[0] == ray_angles.shape[0]
    ), "The number of rays must be the same."

    # Extract x and y coordinates
    x = ray_origins[:, 0]
    y = ray_origins[:, 1]

    # Calculate the end points of the rays
    x_end = x + length * np.cos(ray_angles)
    y_end = y + length * np.sin(ray_angles)

    # Set colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(x)))

    # Plot each ray
    for i in range(len(x)):
        plt.arrow(
            x[i],
            y[i],
            x_end[i] - x[i],
            y_end[i] - y[i],
            head_width=0.3,
            head_length=0.5,
            color=colors[i],
            label=f"{i}",
        )
        # Plot a dot at the base of the arrow
        plt.scatter(x[i], y[i], color=colors[i], s=50)


def plot_location_pred_vs_gt(
    predictions: torch.Tensor,
    groundtruth: torch.Tensor,
    ap_locations: torch.Tensor = None,
    ap_orientations: torch.Tensor = None,
    x_max: float = None,
    y_max: float = None,
) -> plt.Figure:
    """
    Plots location predictions and ground truth as a scatter plot.

    Args:
        predictions: Tensor of shape (n_sample, 2) containing the predicted x and y coordinates.
        groundtruth: Tensor of shape (n_sample, 2) containing the ground truth x and y coordinates.
        ap_locations: Tensor of shape (n_ap, 2) containing the x and y coordinates of the APs.
        ap_orientations: Tensor of shape (n_ap,) containing the orientations of the APs in radians.
        x_max: Maximum x coordinate for the plot. Defaults to 15.
        y_max: Maximum y coordinate for the plot. Defaults to 10.
    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """
    assert predictions.shape == groundtruth.shape, "Predictions and ground truth must have the same shape."

    # Move tensors to CPU if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu()
    if isinstance(groundtruth, torch.Tensor):
        groundtruth = groundtruth.cpu()
    if isinstance(ap_locations, torch.Tensor):
        ap_locations = ap_locations.cpu()
    if isinstance(ap_orientations, torch.Tensor):
        ap_orientations = ap_orientations.cpu()

    fig, ax = plt.subplots()
    ax.scatter(predictions[:, 0], predictions[:, 1], color='red', label='Predictions', s=8)
    ax.scatter(groundtruth[:, 0], groundtruth[:, 1], color='blue', label='Ground Truth', s=8)

    if ap_locations is not None:
        for i, (x, y) in enumerate(ap_locations):
            ax.text(x, y, f'AP{i}', fontsize=12, color='black')
            ax.scatter(x, y, color='green')

            # plot AP orientations
            if ap_orientations is not None:
                orientation = -ap_orientations[i]
                dx = np.cos(orientation)
                dy = np.sin(orientation)
                ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, length_includes_head=True, fc='black', ec='black',)

    if x_max is not None:
        ax.set_xlim(0, x_max)

    if y_max is not None:
        ax.set_ylim(0, y_max)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Predictions vs Ground Truth')
    ax.legend()
    plt.close(fig)
    return fig


def plot_aoa_visualization(sample: AoAVisualizationSample, title: str = "", degree=True) -> plt.Figure:
    """Plot the AoA ToF plots for each AP in the sample.

    Args:
        sample: dataclass that hold data to visualize.
        title: title of the plot.
        degree: if True, the angle will be displayed in degree, otherwise in radian.

    Returns:
        plt figure object.
    """
    n_aps = sample.aoa_tof_plots.shape[0]
    image_height, image_width = sample.aoa_tof_plots.shape[1:3]
    max_angle = 90 if degree else np.pi / 2
    angle_unit = "degree" if degree else "radian"

    # extent of the image, note this value is hardcoded for the current dataset.
    # [x_min_meter (lower left), x_max_meter (lower right), y_min (lower left corner), y_max (upper left corner)]
    extent = [0, 40, max_angle, -max_angle]

    # Calculate the figure size based on the image size and number of plots
    fig_width = n_aps * (image_width / 100)
    fig_height = image_height / 100
    fig, axes = plt.subplots(1, n_aps, figsize=(fig_width, fig_height))

    for ap_index in range(n_aps):
        ax = axes[ap_index]

        # plot AOA-ToF image
        viz_image_sample = sample.aoa_tof_plots[ap_index]
        im = ax.imshow(viz_image_sample, cmap='hot', interpolation='nearest', extent=extent, aspect='auto')

        # plot prediction and ground truth AoA
        ax.axhline(y=sample.aoa_gt[ap_index], color='green', linestyle='--', linewidth=4, label='GT')
        ax.axhline(y=sample.aoa_pred[ap_index], color='red', linestyle='--', linewidth=4, label='Pred')

        # Construct the title string based on the presence of confidence
        subplot_title = (f"AP {ap_index}\n"
                         f"AoA GT: {sample.aoa_gt[ap_index].item():.1f}"
                         f", Pred: {sample.aoa_pred[ap_index].item():.1f}")
        if sample.confidence is not None:
            subplot_title += f", Conf: {sample.confidence[ap_index].item():.1f}"

        ax.set_title(subplot_title)
        ax.set_ylabel(f"AoA ({angle_unit})")
        ax.set_xlabel("ToF (meter)")
        fig.colorbar(im, ax=ax)

    # Add an overall title for the plot
    fig.suptitle(f"{title}")

    # Add a single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Adjust layout to prevent overlap
    fig.tight_layout()

    plt.close(fig)
    return fig


def _plot_2d_histogram(data_x: np.ndarray,
                       data_y: np.ndarray,
                       x_min: float,
                       x_max: float,
                       y_min: float,
                       y_max: float,
                       unit: str,
                       x_label: str,
                       y_label: str,
                       title: str) -> plt.Figure:
    """
    Helper function to plot 2D histograms. Note do not call this function directly
    Use plot_aoa_2d_histograms or plot_aoa_error_2d_histograms instead.

    Ref: https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html


    Args:
        data_x: Array of x-axis data values. Shape (n_samples, n_aps).
        data_y: Array of y-axis data values. Shape (n_samples, n_aps).
        x_min: Minimum value for the x-axis.
        x_max: Maximum value for the x-axis.
        y_min: Minimum value for the y-axis.
        y_max: Maximum value for the y-axis.
        unit: Unit of the angles (degrees or radians).
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Title for the plot.

    Returns:
        fig: The figure object containing the plots.
    """
    assert data_x.shape == data_y.shape, "data_x and data_y must have the same shape."
    n_samples, n_ap = data_x.shape
    num_bins = 50  # Number of bins for both axes
    size_per_subplot = num_bins // 10
    fig, axes = plt.subplots(1, n_ap, figsize=(n_ap * size_per_subplot, size_per_subplot))

    for i in range(n_ap):
        ax = axes[i]

        # Define the bins for the histogram using linspace
        x_bins = np.linspace(x_min, x_max, num_bins + 1)
        y_bins = np.linspace(y_min, y_max, num_bins + 1)

        # Create the 2D histogram
        hist, xedges, yedges = np.histogram2d(data_x[:, i], data_y[:, i], bins=[x_bins, y_bins])

        # Plot the 2D histogram
        im = ax.imshow(hist.T, origin='lower', aspect="auto", interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        fig.colorbar(im, ax=ax, label='Count')
        ax.set_title(f"AP {i}")
        ax.set_xlabel(f"{x_label} ({unit})")
        ax.set_ylabel(f"{y_label} ({unit})")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_aoa_gt_vs_pred_2d_histograms(aoa_pred_all: np.ndarray, aoa_gt_all: np.ndarray, plot_in_degrees: bool = True) -> plt.Figure:
    """
    Plot 2D histograms of ground truth and predicted AoA distributions.

    Args:
        aoa_pred_all: Array of predicted AoA values in radians. Shape (n_samples, n_aps). Plot in x axis.
        aoa_gt_all: Array of ground truth AoA values in radians. Shape (n_samples, n_aps). Plot in y axis.
        plot_in_degrees (bool): If True, plot in degrees. If False, plot in radians.

    Returns:
        fig: The figure object containing the plots.
    """
    assert aoa_pred_all.shape == aoa_gt_all.shape, "Predictions and ground truth must have the same shape."
    unit = "degrees" if plot_in_degrees else "radians"
    aoa_pred_all = np.degrees(aoa_pred_all) if plot_in_degrees else aoa_pred_all
    aoa_gt_all = np.degrees(aoa_gt_all) if plot_in_degrees else aoa_gt_all
    max_angle = 90 if plot_in_degrees else np.pi / 2

    return _plot_2d_histogram(data_x=aoa_pred_all,
                              data_y=aoa_gt_all,
                              x_min= -max_angle,
                              x_max= max_angle,
                              y_min= -max_angle,
                              y_max= max_angle,
                              unit=unit,
                              x_label="Prediction AoA",
                              y_label="Ground Truth AoA",
                              title="2D Histogram of GT and Pred AoA Distribution")


def plot_aoa_abs_error_2d_histograms(aoa_error_all_abs: np.ndarray, aoa_gt_all: np.ndarray, plot_in_degrees: bool = True) -> plt.Figure:
    """
    Plot 2D histograms of ground truth and absolute AoA error distributions.

    Args:
        aoa_error_all_abs: Array of absolute AoA error values in radians. Shape (n_samples, n_aps).
        aoa_gt_all: Array of ground truth AoA values in radians. Shape (n_samples, n_aps).
        plot_in_degrees: If True, plot in degrees. If False, plot in radians.

    Returns:
        fig: The figure object containing the plots.
    """
    assert aoa_error_all_abs.shape == aoa_gt_all.shape, "AoA errors and ground truth must have the same shape."
    unit = "degrees" if plot_in_degrees else "radians"
    aoa_error_all_abs = np.degrees(aoa_error_all_abs) if plot_in_degrees else aoa_error_all_abs
    aoa_gt_all = np.degrees(aoa_gt_all) if plot_in_degrees else aoa_gt_all
    max_angle = 90 if plot_in_degrees else np.pi / 2

    return _plot_2d_histogram(data_x=aoa_error_all_abs,
                              data_y=aoa_gt_all,
                              x_min=0,
                              x_max=50 if plot_in_degrees else np.radians(50),
                              y_min=-max_angle,
                              y_max=max_angle,
                              unit=unit,
                              x_label="Absolute AoA Error",
                              y_label="Ground Truth AoA",
                              title="2D Histogram of Absolute AoA Error Distribution")


def plot_aoa_error_histograms(aoa_error_all: np.ndarray,
                              aoa_error_mean_all: Optional[np.ndarray]=None,
                              aoa_error_std_all: Optional[np.ndarray]=None,
                              plot_in_degrees=True) -> plt.Figure:
    """
    Plot histograms of AoA error for each access point (AP). The AoA error can be raw error or absolute error.
    The original error need to be in radians.

    Args:
        aoa_error_all: Array of AoA error values in radians. Shape (n_samples, n_aps).
        aoa_error_mean_all: Array of mean AoA error values in radians for each AP. Shape (n_aps,).
        aoa_error_std_all: Array of standard deviation of AoA error values in radians for each AP. Shape (n_aps,).
        plot_in_degrees: If True, plot in degrees. If False, plot in radians.

    Returns:
        plt figure object of the image.
    """
    n_ap = aoa_error_all.shape[1]
    if aoa_error_mean_all is not None:
        assert len(aoa_error_mean_all) == n_ap, "Mean AoA error must have the same length as the number of APs."
    if aoa_error_std_all is not None:
        assert len(aoa_error_std_all) == n_ap, "Std AoA error must have the same length as the number of APs."

    unit = "degrees" if plot_in_degrees else "radians"
    fig, axes = plt.subplots(1, n_ap, figsize=(n_ap * 5, 5))
    x_lim = [-np.pi/2, np.pi/2]

    if aoa_error_mean_all is None:
        aoa_error_mean_all = np.mean(aoa_error_all, axis=0)

    if aoa_error_std_all is None:
        aoa_error_std_all = np.std(aoa_error_all, axis=0)

    if plot_in_degrees:
        aoa_error_all = np.degrees(aoa_error_all)
        aoa_error_mean_all = np.degrees(aoa_error_mean_all)
        aoa_error_std_all = np.degrees(aoa_error_std_all)
        x_lim = [-90, 90]


    for i in range(n_ap):
        ax = axes[i]
        ax.hist(aoa_error_all[:, i], bins=100, label=f"AP{i}")

        ax.set_title(f"AP {i}. mean: {aoa_error_mean_all[i]:.2f}, std: {aoa_error_std_all[i]:.2f}")
        ax.set_xlabel(f"AoA Error ({unit})")
        ax.set_ylabel("Count")
        ax.set_xlim(x_lim)

    fig.suptitle("AoA error histogram for each AP")

    return fig


def plot_gt_vs_pred_aoa(aoa_pred_all: np.ndarray, aoa_gt_all: np.ndarray, delta=None, plot_in_degrees=True) -> plt.Figure:
    """
    Plot the scatter plot of GT vs Pred AoA for each AP. The input aoa_pred_all and aoa_gt_all should be in radians.
    If delta is provided, the percentage of aoa_pred samples within +delta and -delta of aoa_gt will be displayed.

    Args:
        aoa_pred_all: Array of predicted AoA values in radians. Shape (n_samples, n_aps).
        aoa_gt_all: Array of ground truth AoA values in radians. Shape (n_samples, n_aps).
        delta: The threshold for error. If provided, the percentage of error within delta will be displayed.
            unit in degree if plot_in_degrees is True, otherwise in radian.
        plot_in_degrees: If True, plot in degrees. If False, plot in radians.

    Returns:
        plt figure object of the image.
    """
    assert aoa_pred_all.shape == aoa_gt_all.shape, "Predictions and ground truth must have the same shape."
    n_ap = aoa_pred_all.shape[1]
    unit = "degree" if plot_in_degrees else "radian"
    x_limit = 90 if plot_in_degrees else np.pi / 2

    if plot_in_degrees:
        aoa_gt_all = np.degrees(aoa_gt_all)
        aoa_pred_all = np.degrees(aoa_pred_all)

    x_values = np.linspace(-x_limit, x_limit, 100)
    fig, axes = plt.subplots(1, n_ap, figsize=(n_ap * 5, 5))

    for i in range(n_ap):
        ax = axes[i]
        aoa_gt_ap = aoa_gt_all[:, i]
        aoa_pred_ap = aoa_pred_all[:, i]

        # compute percentage of error within delta
        if delta is not None:
            aoa_error_ap_abs = np.abs(aoa_gt_ap - aoa_pred_ap)
            num_within_delta = np.sum(aoa_error_ap_abs < delta)
            percentage_within_delta = num_within_delta / len(aoa_error_ap_abs) * 100
            title = f"AP{i}, pct within {delta} {unit}: {percentage_within_delta:.2f}%"
        else:
            title = f"AP{i}"

        ax.scatter(aoa_gt_ap, aoa_pred_ap, s=5)

        # plot lines for y=x and y=x+delta, y=x-delta
        ax.plot(x_values, x_values, color='red', label="y=x")
        if delta is not None:
            ax.plot(x_values, x_values + delta, color='green', linestyle='--', label=f"y=x+{delta}")
            ax.plot(x_values, x_values - delta, color='green', linestyle='--', label=f"y=x-{delta}")
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel(f"GT AoA ({unit})")
        ax.set_ylabel(f"Pred AoA ({unit})")

    fig.suptitle("GT vs Pred AoA for each AP")
    return fig


def plot_location_error_cdf(location_error_all: np.ndarray) -> plt.Figure:
    """
    Plot the cumulative distribution function (CDF) of location errors.

    Args:
        location_error_all: Array of location errors in meters. Shape (n_samples,).

    Returns:
        plt figure object of the image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute the CDF
    sorted_data = np.sort(location_error_all)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot the CDF
    ax.plot(sorted_data, cdf)
    ax.set_title("Location Error CDF")
    ax.set_xlabel("Location Error (m)")
    ax.set_ylabel("CDF")
    ax.grid()

    # Set y-axis range from 0 to 1 with increments of 0.1
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    return fig
