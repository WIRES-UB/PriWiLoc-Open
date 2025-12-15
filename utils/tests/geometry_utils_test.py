from utils.geometry_utils import cos_angle_sum, sin_angle_sum, cos_angle_diff, sin_angle_diff, wrap_to_pi
import torch
import pytest

@pytest.fixture
def theta_values():
    theta1 = torch.rand(5, 4)
    theta2 = torch.rand(4)

    cos_theta1 = torch.cos(theta1)
    sin_theta1 = torch.sin(theta1)
    cos_theta2 = torch.cos(theta2)
    sin_theta2 = torch.sin(theta2)

    return (theta1, cos_theta1, sin_theta1), (theta2, cos_theta2, sin_theta2)

def test_cos_angle_sum(theta_values):
    (theta1, cos_theta1, sin_theta1), (theta2, cos_theta2, sin_theta2) = theta_values

    # run the function
    cos_sum = cos_angle_sum(cos_theta1, sin_theta1, cos_theta2, sin_theta2)
    cos_sum2 = cos_angle_sum(cos_theta2, sin_theta2, cos_theta1, sin_theta1)

    # expected output
    cos_sum_expected = torch.cos(theta1 + theta2)
    cos_sum_expected2 = torch.cos(theta2 + theta1)
    assert torch.allclose(cos_sum_expected, cos_sum_expected2, atol=1e-6)
    assert torch.allclose(cos_sum, cos_sum_expected, atol=1e-6)
    assert torch.allclose(cos_sum2, cos_sum_expected, atol=1e-6)


def test_sin_angle_sum(theta_values):
    (theta1, cos_theta1, sin_theta1), (theta2, cos_theta2, sin_theta2) = theta_values

    # run the function
    sin_sum = sin_angle_sum(cos_theta1, sin_theta1, cos_theta2, sin_theta2)
    sin_sum2 = sin_angle_sum(cos_theta2, sin_theta2, cos_theta1, sin_theta1)

    # compute expected output
    sin_sum_expected = torch.sin(theta1 + theta2)
    sin_sum_expected2 = torch.sin(theta2 + theta1)
    assert torch.allclose(sin_sum_expected, sin_sum_expected2, atol=1e-6)
    assert torch.allclose(sin_sum, sin_sum_expected, atol=1e-6)
    assert torch.allclose(sin_sum2, sin_sum_expected, atol=1e-6)


def test_cos_angle_diff(theta_values):
    (theta1, cos_theta1, sin_theta1), (theta2, cos_theta2, sin_theta2) = theta_values

    # run the function
    cos_diff = cos_angle_diff(cos_theta1, sin_theta1, cos_theta2, sin_theta2)

    # compute expected output
    cos_diff_expected = torch.cos(theta1 - theta2)

    assert torch.allclose(cos_diff, cos_diff_expected, atol=1e-6)


def test_sin_angle_diff(theta_values):
    (theta1, cos_theta1, sin_theta1), (theta2, cos_theta2, sin_theta2) = theta_values

    # run the function
    sin_diff = sin_angle_diff(cos_theta1, sin_theta1, cos_theta2, sin_theta2)

    # compute expected output
    sin_diff_expected = torch.sin(theta1 - theta2)

    assert torch.allclose(sin_diff, sin_diff_expected, atol=1e-6)


def test_wrap_to_pi():
    x = torch.tensor([0, 2 * torch.pi, 3 * torch.pi / 2, 5 * torch.pi / 2, -2 * torch.pi])
    x_wrapped = wrap_to_pi(x)
    expected_values = torch.tensor([0, 0, -torch.pi / 2, torch.pi / 2, 0])
    assert torch.allclose(x_wrapped, expected_values, atol=1e-5)
