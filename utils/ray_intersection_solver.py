"""Line intersection solver"""

import torch

def solve_2d_ray_intersection(ap_aoas: torch.Tensor, ap_xy: torch.Tensor, theta_pred: torch.Tensor) -> torch.Tensor:
    """
    Solves the intersection point of rays in 2D plane using PyTorch operations.
    
    Args:
        ap_aoas: AP angle of arrival measurements. A tensor of shape (n_ap, 1).
        ap_xy: AP locations in map frame. A tensor of shape (n_ap, 2).
        theta_pred: Predicted angle values. A tensor of shape (n_points, n_ap) or (batch, n_points, n_ap).
        
    Returns:
        The intersection points. A tensor of shape (n_points, 2).
    """
    assert ap_xy.shape[1] == 2, "This function solves the intersection point of rays in 2D plane."
    
    if ap_aoas.dim() == 1:
        ap_aoas = ap_aoas.unsqueeze(-1)
        
    # Handle 3D input by reshaping
    if theta_pred.dim() == 3:
        theta_pred = theta_pred.reshape(theta_pred.shape[0] * theta_pred.shape[1], -1)
    
    n_points, n_ap = theta_pred.shape
    assert theta_pred.shape == (n_points, n_ap), f"Shape mismatch. Expected {(n_points, n_ap)}, but got {theta_pred.shape}."
    assert ap_xy.shape == (n_ap, 2), "Shape mismatch."
    assert ap_aoas.shape == (n_ap, 1), f"Shape mismatch, current shape ap aoas: {ap_aoas.shape}, n-ap = {n_ap}"

    # Convert to PyTorch operations - keep everything as tensors
    device = theta_pred.device
    dtype = theta_pred.dtype
    
    # Ensure all tensors are on the same device and dtype
    ap_xy = ap_xy.to(device=device, dtype=dtype)
    ap_aoas = ap_aoas.to(device=device, dtype=dtype)
    
    # Scale theta_pred by pi/2
    theta_pred_scaled = theta_pred * torch.pi / 2
    
    # Initialize output tensor
    output_xy = torch.zeros(n_points, 2, device=device, dtype=dtype)
    
    # Create identity matrix for weighting
    W = torch.eye(n_ap, device=device, dtype=dtype)
    
    # Process each point
    for i in range(n_points):
        # Compute angles: theta_pred[i, :] + ap_aoas.T
        # ap_aoas.T has shape (1, n_ap), theta_pred[i, :] has shape (n_ap,)
        x = theta_pred_scaled[i, :] + ap_aoas.squeeze()  # Shape: (n_ap,)
        
        # Construct matrix A: [[sin(x)], [-cos(x)]]^T -> (n_ap, 2)
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        A = torch.stack([sin_x, -cos_x], dim=1)  # Shape: (n_ap, 2)
        
        # Construct vector B
        B = ap_xy[:, 0] * sin_x - ap_xy[:, 1] * cos_x  # Shape: (n_ap,)
        
        # Solve weighted least squares: (A^T W A)^{-1} A^T W B
        AtWA = A.T @ W @ A  # Shape: (2, 2)
        AtWB = A.T @ W @ B  # Shape: (2,)
        
        # Solve the linear system
        output_xy[i, :] = torch.linalg.solve(AtWA, AtWB)
    
    return output_xy

def solve_ray_intersection(
    ap_xy: torch.Tensor, theta: torch.Tensor, confidence_score: torch.Tensor = None
) -> torch.Tensor:
    """
    Solves the intersection point of n rays in a 2D plane. The rays start from the locations of the Access Points (APs),
    and their angles (in map frame, w.r.t to horizontal axis) are given by theta, which represents the Angle of Arrival (AoA)
    in the map frame. The confidence score indicates the importance of each AoA measurement in the least squares problem.
    Reference: https://www.notion.so/DLoc-Design-37e43b97634d4c848b237ae06fe08ac4?pvs=4

    Args:
        ap_xy: APs location in map frame. A tensor of shape (n_ap, 2).
        theta: Angle of rays with respect to the horizontal axis, which is AoA of each AP in map frame. A tensor of shape (n_ap,).
        confidence_score: AoA Confidence score that determines the imporance of AoA measurment in least square porblem.
            A tensor of shape (n_ap,).

    Returns:
        The intersection point of the rays. A tensor of shape (2,).
    """
    assert (
        ap_xy.shape[0] == theta.shape[0]
    ), "Shape mismatch."

    if confidence_score is None:
        confidence_score = torch.ones_like(theta)
    else:
        assert (
            ap_xy.shape[0] == confidence_score.shape[0]
        ), "Shape mismatch."
        # Assert that confidence_score is not all zeros
        assert torch.any(confidence_score), "Confidence score tensor must not be all zeros."

    # Compute the elements of the matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Matrix elements
    A11 = torch.sum(confidence_score * (1 - cos_theta**2))
    A12 = -torch.sum(confidence_score * cos_theta * sin_theta)
    A21 = A12
    A22 = torch.sum(confidence_score * (1 - sin_theta**2))

    # Construct the matrix
    A = torch.tensor([[A11, A12], [A21, A22]])

    # Compute the elements of the right-hand side vector
    x_i = ap_xy[:, 0]
    y_i = ap_xy[:, 1]

    b1 = torch.sum(confidence_score * ((1 - cos_theta**2) * x_i - cos_theta * sin_theta * y_i))
    b2 = torch.sum(confidence_score * ((1 - sin_theta**2) * y_i - cos_theta * sin_theta * x_i))

    # Construct the right-hand side vector
    b = torch.tensor([b1, b2])

    # Solve the linear system Ax = b
    solution = torch.linalg.solve(A, b)

    return solution


def solve_ray_intersection_batch(
    ap_xy: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    confidence_score: torch.Tensor
) -> torch.Tensor:
    """
    See docstring from above function for more details. This function performs the same calculation in batch mode.
    Reference: https://www.notion.so/DLoc-Design-37e43b97634d4c848b237ae06fe08ac4?pvs=4

    Args:
        ap_xy: AP locations in map frame. A tensor of shape (n_ap, 2).
        cos_theta: Cos of AoA measurement in map frame. A tensor of shape (batch_size, n_ap).
        sin_theta: Sin of AoA measurement in map frame. A tensor of shape (batch_size, n_ap).
        confidence_score: AoA Confidence score that determines the importance of AoA measurement in least square problem.
            A tensor of shape (batch_size, n_ap).

    Returns:
        The intersection point of the rays formed by AoA from each AP. A tensor of shape (batch_size, 2).
    """
    assert ap_xy.shape[1] == 2, "This function solves the intersection point of rays in 2D plane."
    batch_size = cos_theta.shape[0]
    n_ap = ap_xy.shape[0]
    assert cos_theta.shape == (batch_size, n_ap), f"Shape mismatch. Expected {(batch_size, n_ap)}, but got {cos_theta.shape}."
    assert sin_theta.shape == cos_theta.shape, "Shape mismatch."
    assert confidence_score.shape == cos_theta.shape, "Shape mismatch."
    assert ap_xy.shape == (n_ap, 2), "Shape mismatch."

    # Matrix elements, each element has shape (batch_size, 1)
    A11 = torch.sum(confidence_score * (1 - cos_theta**2), dim=1).unsqueeze(1)
    A12 = -torch.sum(confidence_score * cos_theta * sin_theta, dim=1).unsqueeze(1)
    A21 = A12
    A22 = torch.sum(confidence_score * (1 - sin_theta**2), dim=1).unsqueeze(1)

    # Construct the matrix A, shape is (batch_size, 2, 2)
    A = torch.cat((A11, A12, A21, A22), dim=1).view(batch_size, 2, 2)

    # Compute the elements of the right-hand side vector, each element has shape (batch_size, 1)
    x_i = ap_xy[:, 0] # shape is (n_ap,)
    y_i = ap_xy[:, 1] # shape is (n_ap,)

    b1 = torch.sum(confidence_score * ((1 - cos_theta**2) * x_i - cos_theta * sin_theta * y_i), dim=1).unsqueeze(1)
    b2 = torch.sum(confidence_score * ((1 - sin_theta**2) * y_i - cos_theta * sin_theta * x_i), dim=1).unsqueeze(1)

    # Construct the right-hand side matrix, shape is (batch_size, 2)
    b = torch.cat((b1, b2), dim=1)

    # Solve the linear system Ax = b
    # A.shape = (batch_size, 2, 2), b.shape = (batch_size, 2)
    # solution.shape = (batch_size, 2)
    solution = torch.linalg.solve(A, b)

    return solution


import numpy as np

def solve_ray_intersection_batch_np(
    ap_xy: np.ndarray,
    cos_theta: np.ndarray,
    sin_theta: np.ndarray,
    confidence_score: np.ndarray
) -> np.ndarray:
    """
    Computes the intersection point of rays formed by AoA measurements from multiple APs
    using a weighted least-squares approach, implemented in NumPy.

    Args:
        ap_xy: AP locations in map frame. An array of shape (n_ap, 2).
        cos_theta: Cosine of AoA measurement in map frame. An array of shape (batch_size, n_ap).
        sin_theta: Sine of AoA measurement in map frame. An array of shape (batch_size, n_ap).
        confidence_score: AoA Confidence score that determines the importance of AoA measurement in least square problem.
            An array of shape (batch_size, n_ap).

    Returns:
        The intersection point of the rays formed by AoA from each AP. An array of shape (batch_size, 2).
    """
    assert ap_xy.shape[1] == 2, "This function solves the intersection point of rays in 2D plane."
    batch_size = cos_theta.shape[0]
    n_ap = ap_xy.shape[0]
    assert cos_theta.shape == (batch_size, n_ap), f"Shape mismatch. Expected {(batch_size, n_ap)}, but got {cos_theta.shape}."
    assert sin_theta.shape == cos_theta.shape, "Shape mismatch."
    assert confidence_score.shape == cos_theta.shape, "Shape mismatch."
    assert ap_xy.shape == (n_ap, 2), "Shape mismatch."

    # Ensure confidence_score has the correct dimensions for broadcasting if needed, although asserts should cover this.
    # confidence_score = confidence_score[:, :, np.newaxis] # If confidence had shape (batch_size, n_ap) -> (batch_size, n_ap, 1)

    # Matrix elements, each element will have shape (batch_size, 1) after summing
    # Use keepdims=True to maintain the dimension for concatenation
    A11 = np.sum(confidence_score * (1 - cos_theta**2), axis=1, keepdims=True)
    A12 = -np.sum(confidence_score * cos_theta * sin_theta, axis=1, keepdims=True)
    A21 = A12
    A22 = np.sum(confidence_score * (1 - sin_theta**2), axis=1, keepdims=True)

    # Construct the matrix A for each batch item, shape is (batch_size, 2, 2)
    # np.concatenate stacks along the specified axis (axis=1 here)
    A = np.concatenate((A11, A12, A21, A22), axis=1).reshape(batch_size, 2, 2)

    # Compute the elements of the right-hand side vector
    # Extract x_i and y_i from ap_xy, shapes are (n_ap,)
    x_i = ap_xy[:, 0]
    y_i = ap_xy[:, 1]

    # Broadcasting: arrays of shape (batch_size, n_ap) are multiplied by (n_ap,), result is (batch_size, n_ap)
    b1_terms = confidence_score * ((1 - cos_theta**2) * x_i - cos_theta * sin_theta * y_i)
    b2_terms = confidence_score * ((1 - sin_theta**2) * y_i - cos_theta * sin_theta * x_i)

    # Sum along the n_ap dimension (axis=1)
    b1 = np.sum(b1_terms, axis=1, keepdims=True) # shape (batch_size, 1)
    b2 = np.sum(b2_terms, axis=1, keepdims=True) # shape (batch_size, 1)

    # Construct the right-hand side matrix b, shape is (batch_size, 2)
    b = np.concatenate((b1, b2), axis=1)

    # Solve the linear system Ax = b for each item in the batch
    # np.linalg.solve handles broadcasting correctly for A=(batch_size, 2, 2) and b=(batch_size, 2)
    solution = np.linalg.solve(A, b) # solution.shape = (batch_size, 2)

    return solution

