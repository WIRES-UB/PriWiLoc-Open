import torch
import pytest

from utils.schema import (
    APMetadata,
    DLocBatchDataSample,
    DLocBatchSequenceDataSample,
    DLocDataSample,
)


def _ap_metadata(n_aps: int = 3) -> APMetadata:
    """Create valid AP metadata for schema tests."""

    return APMetadata(
        ap_aoas=torch.zeros(n_aps),
        ap_locs=torch.zeros(n_aps, 2),
    )


def _batched_ap_metadata(batch_size: int, n_aps: int = 3) -> torch.Tensor:
    """Create batched tensor metadata matching collated dataset output."""

    metadata = _ap_metadata(n_aps).to_tensor()
    return metadata.unsqueeze(0).repeat(batch_size, 1, 1)


def test_dloc_data_sample() -> None:
    """Test DLocDataSample class."""
    # Create a sample with valid data
    sample = DLocDataSample(
        features_2d=torch.randn(3, 64, 64),
        aoa_label=torch.randn(3,),
        location_label=torch.rand(2,),
        velocity=torch.randn(2,),
        timestamps=torch.randn(1,),
        ap_metadata=_ap_metadata(),
    )
    assert not sample.is_empty()
    assert not sample.has_nan()

    # Create a sample with empty data
    empty_sample = DLocDataSample(
        features_2d=torch.empty(0),
        aoa_label=torch.empty(0),
        location_label=torch.empty(0),
        velocity=torch.empty(0),
        timestamps=torch.empty(0),
        ap_metadata=_ap_metadata(),
    )
    assert empty_sample.is_empty()

    # Create a sample with NaN values
    nan_sample = DLocDataSample(
        features_2d=torch.tensor([[1.0, 2.0], [3.0, float('nan')]]),
        aoa_label=torch.tensor([0.1, 0.2]),
        location_label=torch.tensor([1.0, 2.0]),
        velocity=torch.tensor([0.5, 0.5]),
        timestamps=torch.tensor([1.0]),
        ap_metadata=_ap_metadata(),
    )
    assert nan_sample.has_nan()


def test_dloc_sample_batch() -> None:
    """Test DLocBatchDataSample class.
    """
    # Create a batch with valid data.
    batch = DLocBatchDataSample(
        features_2d=torch.randn(10, 3, 64, 64),
        aoa_label=torch.randn(10, 3),
        location_label=torch.rand(10, 2),
        velocity=torch.randn(10, 2),
        timestamps=torch.randn(10, 1),
        ap_metadata=_batched_ap_metadata(10),
        rssi=torch.randn(10, 3),
    )
    assert batch.get_batch_size() == 10
    assert not batch.is_empty()
    assert not batch.has_nan()

    # Test unmatched dimensions.
    with pytest.raises(ValueError):
        DLocBatchDataSample(
            features_2d=torch.randn(10, 3, 64, 64),
            aoa_label=torch.randn(5, 3),
            location_label=torch.rand(10, 2),
            velocity=torch.randn(10, 2),
            timestamps=torch.randn(10, 1),
            ap_metadata=_ap_metadata(),
        )

    # Test query support set partition.
    support_set, query_set = batch.partition_support_query_set(support_set_pct=0.8)
    assert support_set.get_batch_size() == 8
    assert query_set.get_batch_size() == 2
    assert support_set.ap_metadata.shape[0] == 8
    assert query_set.ap_metadata.shape[0] == 2
    assert support_set.rssi.shape[0] == 8
    assert query_set.rssi.shape[0] == 2
    assert isinstance(support_set, DLocBatchDataSample)
    assert isinstance(query_set, DLocBatchDataSample)


def test_dloc_batch_sequenced_sample() -> None:
    """Test DLocBatchSequenceDataSample class."""
    # Create a sequenced batch with valid data.
    batch = DLocBatchSequenceDataSample(
        features_2d=torch.randn(10, 5, 3, 64, 64),  # 10 samples, sequence length 5
        aoa_label=torch.randn(10, 5, 3),
        location_label=torch.rand(10, 5, 2),
        velocity=torch.randn(10, 5, 2),
        timestamps=torch.randn(10, 5, 1),
        ap_metadata=_batched_ap_metadata(10),
        rssi=torch.randn(10, 3),
    )
    assert batch.get_batch_size() == 10
    assert batch.get_sequence_size() == 5
    assert not batch.is_empty()
    assert not batch.has_nan()

    # Test unmatched dimensions.
    with pytest.raises(ValueError):
        DLocBatchSequenceDataSample(
            features_2d=torch.randn(10, 5, 3, 64, 64),
            aoa_label=torch.randn(10, 5, 3),  # Mismatched dimensions
            location_label=torch.rand(12, 5, 2),
            velocity=torch.randn(10, 5, 2),
            timestamps=torch.randn(10, 5, 1),
            ap_metadata=_ap_metadata(),
        )

    # Test query support set partition.
    support_set, query_set = batch.partition_support_query_set(support_set_pct=0.8)
    assert support_set.get_batch_size() == 8
    assert query_set.get_batch_size() == 2
    assert support_set.ap_metadata.shape[0] == 8
    assert query_set.ap_metadata.shape[0] == 2
    assert support_set.rssi.shape[0] == 8
    assert query_set.rssi.shape[0] == 2
    assert isinstance(support_set, DLocBatchSequenceDataSample)
    assert isinstance(query_set, DLocBatchSequenceDataSample)
