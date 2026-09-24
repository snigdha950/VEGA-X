import json
from pathlib import Path

import numpy as np
import pytest

import src.engine as engine_module
from src.engine import DEMO_POLICIES, POLICIES, SELECTED_POLICY, run_policy
from src.evaluator import evaluate
from src.learner import CUSUMUCB
from src.periodic import ConfidenceGatedTemporalPredictor
from src.receiver import Observation, Receiver
from src.scheduler import MaxAgeGuard
from src.simulator import SCENARIOS, make_scenario
from src.tsrd import inspect_tsrd, load_tsrd_replay


ROOT = Path(__file__).resolve().parents[1]


def test_only_selected_learner_and_display_baselines_are_exposed():
    assert SELECTED_POLICY == "rafts_cusum_ucb"
    assert DEMO_POLICIES == (
        "sequential",
        "random",
        "coverage_sweep",
        "rafts_cusum_ucb",
    )
    assert "rafts_ts" not in POLICIES
    assert "rafts_klucb" not in POLICIES
    assert "rafts_linucb" not in POLICIES


def test_cusum_ucb_rejects_non_binary_feedback():
    learner = CUSUMUCB(4)
    with pytest.raises(ValueError, match="HIT=1 or MISS=0"):
        learner.update(0, 0, 2)


def test_cusum_ucb_scores_are_finite_and_bounded():
    learner = CUSUMUCB(5)
    for slot, observation in enumerate([1, 0, 1, 1, 0, 0]):
        learner.update(slot % 2, slot, observation)
    scores = learner.sample_scores(7)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))


def test_cusum_detects_a_large_regime_change():
    learner = CUSUMUCB(1, drift=0.0, threshold=2.0)
    for slot in range(20):
        learner.update(0, slot, 0)
    for slot in range(20, 40):
        learner.update(0, slot, 1)
        if learner.change_count[0]:
            break
    assert learner.change_count[0] >= 1


def test_controlled_simulator_covers_all_declared_behavior_families():
    assert {
        "periodic",
        "random",
        "bursty",
        "changing",
        "agile",
        "jittered_periodic",
        "spatial_agile",
        "no_emitters",
        "all_active",
        "dense",
        "quiet_to_active",
        "active_to_silent",
        "periodic_to_random",
        "sync_trap",
    } == set(SCENARIOS)
    for scenario in SCENARIOS:
        truth = make_scenario(scenario, slots=12, bands=6, seed=9)
        assert truth.shape == (12, 6)
        assert truth.dtype == np.bool_


def test_retune_dead_time_counts_every_lost_receiver_slot():
    truth = np.zeros((8, 2), dtype=bool)
    result = run_policy(
        truth,
        policy="sequential",
        num_receivers=1,
        retune_delay_slots=3,
        max_age=8,
        pfa=0,
    )
    assert result["retunes"] == 2
    assert result["retune_dead_slots"] == 6
    assert result["observations"] == 2
    assert result["receiver_time_slots"] == 8


def test_sensor_noise_is_common_for_matching_receiver_band_and_slot():
    first = Receiver(0, pd=0.5, pfa=0.5, seed_offset=9)
    second = Receiver(0, pd=0.5, pfa=0.5, seed_offset=9)
    assert first.start_assignment(3, 4)
    assert second.start_assignment(3, 4)
    assert first.observe(True, 4).detected == second.observe(True, 4).detected
    with pytest.raises(ValueError, match="sensitivity_db"):
        Receiver(0, sensitivity_db=float("nan"))


@pytest.mark.parametrize(
    "truth",
    [
        np.zeros((120, 20), dtype=bool),
        np.ones((120, 20), dtype=bool),
        np.eye(20, dtype=bool)[np.arange(120) % 20],
    ],
)
def test_selected_policy_handles_edge_replays_without_coverage_violations(truth):
    result = run_policy(truth, policy=SELECTED_POLICY, seed_offset=3)
    assert result["coverage_violations"] == 0
    assert result["max_observed_age"] <= 80
    assert np.isfinite(result["runtime_seconds"])


def test_compatibility_aliases_run_the_selected_learner():
    truth = np.zeros((80, 20), dtype=bool)
    truth[::5, 3] = True
    selected = run_policy(truth, policy="rafts_cusum_ucb", seed_offset=11)
    alias = run_policy(truth, policy="rafts", seed_offset=11)
    for key in (
        "interception_ratio",
        "capped_mean_tti",
        "retunes",
        "retune_dead_slots",
        "learner_change_detections",
    ):
        assert selected[key] == alias[key]


def test_trace_exposes_change_counts_without_affecting_scheduler_input():
    truth = np.zeros((20, 6), dtype=bool)
    result = run_policy(
        truth,
        policy=SELECTED_POLICY,
        seed_offset=2,
        num_receivers=2,
        max_age=10,
        collect_trace=True,
    )
    trace = result["_trace"]
    assert len(trace) == len(truth)
    assert trace[-1]["change_count_by_band"].shape == (truth.shape[1],)
    assert int(trace[-1]["change_count_by_band"].sum()) == trace[-1][
        "change_detections"
    ]


def test_invalid_learner_output_uses_logged_deadline_safe_fallback(monkeypatch):
    class InvalidLearner:
        def __init__(self, bands):
            self.bands = bands
            self.change_count = np.zeros(bands, dtype=int)

        def sample_scores(self, slot):
            del slot
            return np.full(self.bands, np.nan)

        def posterior_mean(self, slot=None):
            del slot
            return np.full(self.bands, 0.5)

        def update(self, band, slot, observation):
            del band, slot, observation

    monkeypatch.setattr(
        engine_module,
        "make_selected_learner",
        lambda bands, seed: InvalidLearner(bands),
    )
    truth = make_scenario("dense", slots=80, bands=8, seed=2)
    result = run_policy(
        truth,
        policy=SELECTED_POLICY,
        seed_offset=3,
        num_receivers=2,
        max_age=20,
        collect_trace=True,
    )
    assert result["fallback_count"] > 0
    assert result["coverage_violations"] == 0
    assert any(state["fallback"] for state in result["_trace"])
    assert all(
        not state["fallback"] or state["fallback_reason"]
        for state in result["_trace"]
    )


def test_unique_episode_and_prediction_metrics_are_distinct_and_defined():
    truth = np.zeros((10, 2), dtype=bool)
    truth[1:4, 0] = True
    truth[5:7, 0] = True
    observations = [
        Observation(1, 0, 0, 1, 1, 0, 0.9),
        Observation(2, 0, 0, 1, 1, 0, 0.8),
        Observation(3, 0, 0, 1, 1, 0, 0.7),
        Observation(5, 0, 0, 1, 1, 0, 0.9),
        Observation(8, 0, 1, 0, 1, 1, 0.6),
    ]
    result = evaluate(truth, observations, 0, 0, 0, 0)
    assert result["episodes"] == 2
    assert result["intercepted_episodes"] == 2
    assert result["unique_episode_reward"] == 2
    assert result["net_true_reward"] == 3
    assert result["unique_episode_reward_per_receiver_time"] == pytest.approx(0.4)
    assert result["net_true_reward_per_receiver_time"] == pytest.approx(0.6)
    assert np.isfinite(result["learner_prediction_accuracy"])
    assert np.isfinite(result["learner_brier_score"])


def test_temporal_predictor_coalesces_duplicate_hit_slots():
    predictor = ConfidenceGatedTemporalPredictor(num_bands=1, min_hits=4)
    for slot in (2, 2, 6, 10, 14):
        predictor.update(0, slot, 1)
    bonus, confidence = predictor.evidence(0, 18)
    assert bonus > 0
    assert confidence == pytest.approx(1.0)


def test_all_four_tsrd_files_are_valid_and_replayable():
    names = ("config_0.h5", "config_1.h5", "config_10.h5", "config_103.h5")
    for name in names:
        path = ROOT / "data/tsrd" / name
        info = inspect_tsrd(path)
        assert info["shape"][0] > 0
        replay = load_tsrd_replay(path, num_slots=12, num_bands=20)
        assert replay.truth.shape == (12, 20)


def test_frozen_evidence_matches_selected_policy():
    report = json.loads(
        (ROOT / "evidence/selection_report.json").read_text(encoding="utf-8")
    )
    selection = report["selection"]
    assert selection["recommended_policy"] == SELECTED_POLICY
    assert selection["recommended_label"] == "RAFTS-CUSUM-UCB"


def test_invalid_truth_and_infeasible_configuration_fail_cleanly():
    with pytest.raises(ValueError, match="two-dimensional"):
        run_policy(np.zeros(20, dtype=bool))
    with pytest.raises(ValueError, match="infeasible"):
        run_policy(
            np.zeros((20, 20), dtype=bool),
            num_receivers=1,
            max_age=10,
            retune_delay_slots=1,
        )


def test_max_age_guard_and_band_scaling_are_explicit():
    guard = MaxAgeGuard(20, 2, max_age=40, retune_delay=1)
    guard.update_observation(0, 3)
    assert guard.ages(4)[0] == 1
    for bands in (20, 100, 500):
        required_age = int(np.ceil(bands / 2)) * 2
        result = run_policy(
            np.zeros((4, bands), dtype=bool),
            policy=SELECTED_POLICY,
            max_age=required_age,
        )
        assert result["observations"] > 0


def test_inflight_targets_and_joint_assignments_remain_distinct():
    truth = make_scenario("dense", slots=80, bands=8, seed=5)
    result = run_policy(
        truth,
        policy=SELECTED_POLICY,
        seed_offset=4,
        max_age=20,
        retune_delay_slots=3,
        collect_trace=True,
    )
    for state in result["_trace"]:
        assigned = [assignment["band"] for assignment in state["assignments"]]
        assert len(assigned) == len(set(assigned))
        assert set(assigned).isdisjoint(state["inflight_bands"])
    assert result["duplicate_assignments"] == 0
    assert result["coverage_violations"] == 0
