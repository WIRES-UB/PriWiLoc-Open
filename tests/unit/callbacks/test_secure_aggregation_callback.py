"""Tests for secure plotting, audit export, and bounded checkpoints."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pytest
import torch

from callbacks.secure_aggregation_callback import SecureAggregationCallback
from models.federated_learning_secagg import SecureFederatedLearningModel
from models.secure_agg.trust_tracking import (
    APTrustTracker,
    ServerWeightChangeTracker,
)


class _Aggregator:
    def __init__(self):
        self.weight_change_tracker = ServerWeightChangeTracker()


class _Model:
    def __init__(self):
        self.ap_trust_trackers = [APTrustTracker(0), APTrustTracker(1)]
        self.secure_aggregator = _Aggregator()

    @property
    def server_weight_change_history(self):
        return self.secure_aggregator.weight_change_tracker.history

    def get_ap_trust_history(self):
        return {
            tracker.ap_id: tracker.history
            for tracker in self.ap_trust_trackers
        }


class _Trainer:
    is_global_zero = True
    logger = None
    global_step = 8
    current_epoch = 0


def _record_round(model: _Model) -> None:
    for tracker, trusted in zip(model.ap_trust_trackers, (True, False)):
        tracker.record(
            epoch=0,
            round_idx=0,
            step_idx=8,
            trusted_weights=trusted,
        )
    model.secure_aggregator.weight_change_tracker.record(
        epoch=0,
        round_idx=0,
        step_idx=8,
        changed_ap_ids=[1],
    )


def test_audit_export_is_immutable_and_drains_only_after_success(tmp_path):
    model = _Model()
    _record_round(model)
    callback = SecureAggregationCallback(output_dir=str(tmp_path))

    path = callback._export_events(_Trainer(), model)

    assert path is not None
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event_type"] for record in records] == [
        "ap_trust",
        "ap_trust",
        "server_response",
    ]
    assert all(not tracker.history for tracker in model.ap_trust_trackers)
    assert not model.server_weight_change_history
    trust_history, server_history = callback._complete_history(model)
    assert [len(events) for events in trust_history.values()] == [1, 1]
    assert len(server_history) == 1


def test_resumed_callback_starts_a_linked_audit_segment(tmp_path):
    original = SecureAggregationCallback(output_dir=str(tmp_path))
    restored = SecureAggregationCallback(output_dir=str(tmp_path))

    restored.load_state_dict(original.state_dict())

    assert restored.parent_segment_id == original.segment_id
    assert restored.segment_lineage == (
        (original.segment_id, 0, str(tmp_path.resolve())),
    )
    assert restored.segment_id != original.segment_id
    assert restored.export_index == 0


def test_resume_lineage_stops_at_the_checkpoint_export_boundary(tmp_path):
    model = _Model()
    callback = SecureAggregationCallback(output_dir=str(tmp_path))
    _record_round(model)
    callback._export_events(_Trainer(), model)
    checkpoint_state = callback.state_dict()

    _record_round(model)
    callback._export_events(_Trainer(), model)
    resumed = SecureAggregationCallback(output_dir=str(tmp_path))
    resumed.load_state_dict(checkpoint_state)

    trust_history, server_history = resumed._complete_history(model)

    assert [len(events) for events in trust_history.values()] == [1, 1]
    assert len(server_history) == 1


def test_failed_audit_export_keeps_pending_events(tmp_path):
    model = _Model()
    _record_round(model)
    callback = SecureAggregationCallback(output_dir=str(tmp_path))
    collision = tmp_path / (
        f"audit_{callback.segment_id}_{callback.export_index:06d}.jsonl"
    )
    collision.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        callback._export_events(_Trainer(), model)

    assert all(tracker.history for tracker in model.ap_trust_trackers)
    assert model.server_weight_change_history


def test_checkpoint_callback_state_uses_post_export_boundary(tmp_path):
    model = _Model()
    _record_round(model)
    callback = SecureAggregationCallback(output_dir=str(tmp_path))
    checkpoint = {"callbacks": {}}

    callback.on_save_checkpoint(_Trainer(), model, checkpoint)

    saved = checkpoint["callbacks"][callback.state_key]
    assert saved["export_index"] == 1
    assert saved["output_dir"] == str(tmp_path.resolve())


def test_figures_close_when_logger_fails(monkeypatch, tmp_path):
    model = _Model()
    _record_round(model)
    callback = SecureAggregationCallback(output_dir=str(tmp_path))
    before = set(plt.get_fignums())

    def fail(*args, **kwargs):
        raise RuntimeError("logger failed")

    monkeypatch.setattr(
        "callbacks.secure_aggregation_callback.log_figure",
        fail,
    )
    with pytest.raises(RuntimeError, match="logger failed"):
        callback._log_figures(_Trainer(), model)

    assert set(plt.get_fignums()) == before


def test_model_checkpoint_contains_counters_but_no_event_histories():
    class Holder:
        _secure_round_idx = torch.tensor(4)
        _secure_rejections_cumulative = torch.tensor(2)
        ap_trust_trackers = [APTrustTracker(0)]

    holder = Holder()
    holder.ap_trust_trackers[0].record(
        epoch=0,
        round_idx=0,
        trusted_weights=True,
    )
    checkpoint = {}

    SecureFederatedLearningModel.on_save_checkpoint(holder, checkpoint)

    state = checkpoint["secure_resume_state"]
    assert state["round_idx"] == 4
    assert state["rejections_cumulative"] == 2
    assert state["aps"] == [{"ap_id": 0, "correct": 1, "incorrect": 0}]
    assert "history" not in repr(state)

    restored = Holder()
    restored._secure_round_idx = torch.tensor(0)
    restored._secure_rejections_cumulative = torch.tensor(0)
    restored.ap_trust_trackers = [APTrustTracker(0)]
    SecureFederatedLearningModel.on_load_checkpoint(restored, checkpoint)

    assert restored._secure_round_idx.item() == 4
    assert restored._secure_rejections_cumulative.item() == 2
    assert restored.ap_trust_trackers[0].correct == 1
    assert restored.ap_trust_trackers[0].history == ()
