import torch
import pytest

from utils.schema import DLocDataSample, DLocBatchDataSample, DLocBatchSequencedDataSample

def test_dloc_data_sample() -> None:
    """Test DLocDataSample class."""
    # Create a sample with valid data
    sample = DLocDataSample(
        features_2d=torch.randn(3, 64, 64),
        aoa_label=torch.randn(3,),
        location_label=torch.rand(2,),
        velocity=torch.randn(2,),
        timestamps=torch.randn(1,)
    )
    assert not sample.is_empty()
    assert not sample.has_nan()

    # Create a sample with empty data
    empty_sample = DLocDataSample(
        features_2d=torch.empty(0),
        aoa_label=torch.empty(0),
        location_label=torch.empty(0),
        velocity=torch.empty(0),
        timestamps=torch.empty(0)
    )
    assert empty_sample.is_empty()

    # Create a sample with NaN values
    nan_sample = DLocDataSample(
        features_2d=torch.tensor([[1.0, 2.0], [3.0, float('nan')]]),
        aoa_label=torch.tensor([0.1, 0.2]),
        location_label=torch.tensor([1.0, 2.0]),
        velocity=torch.tensor([0.5, 0.5]),
        timestamps=torch.tensor([1.0])
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
        timestamps=torch.randn(10, 1)
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
            timestamps=torch.randn(10, 1)
        )

    # Test query support set partition.
    support_set, query_set = batch.partition_support_query_set(support_set_pct=0.8)
    assert support_set.get_batch_size() == 8
    assert query_set.get_batch_size() == 2
    assert isinstance(support_set, DLocBatchDataSample)
    assert isinstance(query_set, DLocBatchDataSample)


def test_dloc_batch_sequenced_sample() -> None:
    """Test DLocBatchSequencedDataSample class."""
    # Create a sequenced batch with valid data.
    batch = DLocBatchSequencedDataSample(
        features_2d=torch.randn(10, 5, 3, 64, 64),  # 10 samples, sequence length 5
        aoa_label=torch.randn(10, 5, 3),
        location_label=torch.rand(10, 5, 2),
        velocity=torch.randn(10, 5, 2),
        timestamps=torch.randn(10, 5, 1)
    )
    assert batch.get_batch_size() == 10
    assert batch.get_sequence_length() == 5
    assert not batch.is_empty()
    assert not batch.has_nan()

    # Test unmatched dimensions.
    with pytest.raises(ValueError):
        DLocBatchSequencedDataSample(
            features_2d=torch.randn(10, 5, 3, 64, 64),
            aoa_label=torch.randn(10, 5, 3),  # Mismatched dimensions
            location_label=torch.rand(12, 5, 2),
            velocity=torch.randn(10, 5, 2),
            timestamps=torch.randn(10, 5, 1)
        )

    # Test query support set partition.
    support_set, query_set = batch.partition_support_query_set(support_set_pct=0.8)
    assert support_set.get_batch_size() == 8
    assert query_set.get_batch_size() == 2
    assert isinstance(support_set, DLocBatchSequencedDataSample)
    assert isinstance(query_set, DLocBatchSequencedDataSample)
