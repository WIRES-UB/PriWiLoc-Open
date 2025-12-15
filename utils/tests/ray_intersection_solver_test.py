import numpy as np
import torch

from utils.ray_intersection_solver import solve_ray_intersection, solve_ray_intersection_batch


def test_solve_ray_intersection():
    """Test function solve_ray_intersection."""
    # Example input data
    ap_xy = torch.Tensor([[0, 0], [3, 4], [8, 2]])
    ap_aoa = torch.Tensor([np.pi / 4, np.pi / 6, 3 * np.pi / 4])

    # Expected output
    expected_output = torch.Tensor([4.9665, 5.0580])

    # Call the function
    intersection_point = solve_ray_intersection(ap_xy, ap_aoa)

    # Assert the result is as expected
    assert torch.allclose(intersection_point, expected_output, atol=1e-4)

    # Test with confidence scores
    confidence_score = torch.Tensor([1, 0, 1])
    intersection_point_with_conf = solve_ray_intersection(ap_xy, ap_aoa, confidence_score)
    expected_with_conf = torch.Tensor([5, 5])
    assert torch.allclose(intersection_point_with_conf, expected_with_conf, atol=1e-4)

def test_solve_ray_intersection_batch():
    """Test function solve_ray_intersection_batch."""
    # Example input data
    batch_size = 4
    n_ap = 6
    ap_xy = torch.rand((n_ap, 2))
    theta_batch = torch.rand((batch_size, n_ap))
    cos_theta_batch = torch.cos(theta_batch)
    sin_theta_batch = torch.sin(theta_batch)
    confidence_scores_batch = torch.rand((batch_size, n_ap))

    # Call the function to calculate the intersection point entire batch
    intersection_points = solve_ray_intersection_batch(ap_xy, cos_theta_batch, sin_theta_batch, confidence_scores_batch)

    # Assert the result is as expected
    for batch_idx in range(batch_size):
        expected_output = solve_ray_intersection(ap_xy, theta_batch[batch_idx], confidence_scores_batch[batch_idx])
        assert torch.allclose(expected_output, intersection_points[batch_idx], atol=1e-4)
