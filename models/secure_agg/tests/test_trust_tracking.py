import pytest
import torch

from models.aggregation import ClientUpdate
from models.federated_learning import FederatedLearningModel
from models.secure_agg.attacks import AttackRule, ScalingAttack
from models.secure_agg.selectors import (
    APSelector,
    ParameterSelector,
    RoundSelector,
)
from models.secure_agg.aggregator import SecureAggregator
from models.secure_agg.trust_tracking import (
    APTrustTracker,
    ServerWeightChangeTracker,
)
from utils.secure_agg_plotting import (
    create_ap_binary_trust_figure,
    create_ap_cumulative_trust_figure,
    create_server_attack_figure,
)
from utils.experiment_logging import log_figure


def _client_states():
    return [
        ClientUpdate(0, {"weight": torch.tensor([1.0, 2.0])}),
        ClientUpdate(1, {"weight": torch.tensor([3.0, 4.0])}),
        ClientUpdate(2, {"weight": torch.tensor([5.0, 6.0])}),
    ]


def test_server_can_change_none_some_or_all_ap_responses():
    aggregator = SecureAggregator(
        frac_bits=8,
        modulus_bits=31,
        k=2,
        master_seed=7,
        attack_rules=(
            AttackRule(
                name="round_one",
                attack=ScalingAttack(factor=2),
                ap_selector=APSelector.from_value([1]),
                rounds=RoundSelector.from_value([1]),
                parameter_selector=ParameterSelector(),
            ),
            AttackRule(
                name="round_two",
                attack=ScalingAttack(factor=2),
                ap_selector=APSelector.from_value("all"),
                rounds=RoundSelector.from_value([2]),
                parameter_selector=ParameterSelector(),
            ),
        ),
    )

    responses, report = aggregator.aggregate(
        _client_states(), round_idx=0, epoch_idx=0
    )
    assert report.ap_trust == (True, True, True)
    assert report.changed_ap_ids == ()
    assert report.step == 0
    assert report.accepted
    assert torch.equal(
        responses[0].state["weight"],
        torch.tensor([3.0, 4.0]),
    )

    responses, report = aggregator.aggregate(
        _client_states(), round_idx=1, epoch_idx=0
    )
    assert report.ap_trust == (True, False, True)
    assert report.changed_ap_ids == (1,)
    assert not report.accepted
    assert responses[0].verified
    assert not responses[1].verified

    responses, report = aggregator.aggregate(
        _client_states(), round_idx=2, epoch_idx=1
    )
    assert report.ap_trust == (False, False, False)
    assert report.changed_ap_ids == (0, 1, 2)
    assert not report.accepted
    assert not any(response.verified for response in responses)

    assert [
        (event.epoch, event.round, event.changed_ap_ids)
        for event in aggregator.weight_change_history
    ] == [
        (0, 0, ()),
        (0, 1, (1,)),
        (1, 2, (0, 1, 2)),
    ]


def test_each_ap_tracker_maintains_binary_and_cumulative_trust():
    tracker = APTrustTracker(ap_id=2)
    first = tracker.record(epoch=0, round_idx=0, trusted_weights=True)
    second = tracker.record(epoch=0, round_idx=1, trusted_weights=False)
    third = tracker.record(epoch=1, round_idx=2, trusted_weights=True)

    assert [event.trusted_weights for event in tracker.history] == [1, 0, 1]
    assert first.trust_score == 1.0
    assert second.trust_score == 0.5
    assert third.trust_score == pytest.approx(2 / 3)
    assert tracker.correct == 2
    assert tracker.incorrect == 1

    restored = APTrustTracker(ap_id=2)
    restored.load_state_dict(tracker.state_dict())
    assert restored.history == ()
    assert restored.trust_score == pytest.approx(2 / 3)


def test_server_tracker_exposes_immutable_events_and_drains_after_export():
    tracker = ServerWeightChangeTracker()
    tracker.record(epoch=0, round_idx=0, changed_ap_ids=[])
    tracker.record(epoch=1, round_idx=1, changed_ap_ids=[2, 0, 2])

    snapshot = tracker.history
    assert isinstance(snapshot, tuple)
    assert tracker.drain_history(1) == snapshot[:1]
    assert tracker.history == snapshot[1:]


def test_only_trusting_aps_load_the_global_state():
    class EncoderHolder:
        resnet_encoder_list = [
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.Linear(1, 1, bias=False),
        ]

    holder = EncoderHolder()
    for ap_id, encoder in enumerate(holder.resnet_encoder_list):
        encoder.weight.data.fill_(float(ap_id))

    FederatedLearningModel.set_resnet_encoder_parameters(
        holder,
        {"weight": torch.tensor([[9.0]])},
        ap_ids=[0, 2],
    )

    assert holder.resnet_encoder_list[0].weight.item() == 9.0
    assert holder.resnet_encoder_list[1].weight.item() == 1.0
    assert holder.resnet_encoder_list[2].weight.item() == 9.0


def test_comet_figures_use_training_steps_and_stacked_attack_axes():
    trackers = [APTrustTracker(ap_id) for ap_id in range(4)]
    decisions = (
        (True, True, False, True),
        (True, False, True, True),
    )
    for round_idx, step_idx in enumerate((10, 20)):
        for tracker, trusted in zip(trackers, decisions[round_idx]):
            tracker.record(
                epoch=round_idx,
                round_idx=round_idx,
                step_idx=step_idx,
                trusted_weights=trusted,
            )

    server = ServerWeightChangeTracker()
    server.record(
        epoch=0,
        round_idx=0,
        step_idx=10,
        changed_ap_ids=[2],
    )
    server.record(
        epoch=1,
        round_idx=1,
        step_idx=20,
        changed_ap_ids=[1],
    )

    history = {tracker.ap_id: tracker.history for tracker in trackers}
    binary_figure = create_ap_binary_trust_figure(history)
    score_figure = create_ap_cumulative_trust_figure(history)
    attack_figure = create_server_attack_figure(server.history, n_aps=4)

    assert len(binary_figure.axes) == 1
    assert len(binary_figure.axes[0].lines) == 4
    assert list(binary_figure.axes[0].lines[0].get_xdata()) == [10, 20]
    assert len(score_figure.axes[0].lines) == 4
    assert len(attack_figure.axes) == 4
    assert list(attack_figure.axes[1].lines[0].get_ydata()) == [0, 1]
    assert list(attack_figure.axes[2].lines[0].get_ydata()) == [1, 0]
    shared_x_axes = attack_figure.axes[0].get_shared_x_axes()
    assert all(
        shared_x_axes.joined(attack_figure.axes[0], axis)
        for axis in attack_figure.axes[1:]
    )


def test_comet_figure_helper_passes_the_current_step():
    class Experiment:
        def __init__(self):
            self.calls = []

        def log_figure(self, **kwargs):
            self.calls.append(kwargs)

    class Logger:
        experiment = Experiment()

    figure = create_ap_binary_trust_figure({0: APTrustTracker(0).history})
    log_figure(Logger(), "trust", figure, step=42, overwrite=True)

    assert Logger.experiment.calls[0]["figure_name"] == "trust"
    assert Logger.experiment.calls[0]["step"] == 42
    assert Logger.experiment.calls[0]["overwrite"] is True
