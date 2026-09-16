# SPDX-License-Identifier: AGPL-3.0-only

from routes import self_training


def test_fixed_holdout_scoring_is_external_to_the_model():
    arithmetic = self_training._HOLDOUT_BENCHMARK[0]
    json_contract = self_training._HOLDOUT_BENCHMARK[1]
    ordered = self_training._HOLDOUT_BENCHMARK[2]
    assert self_training._score_holdout(arithmetic, "391") == 1.0
    assert self_training._score_holdout(arithmetic, "The answer is 391") == 1.0
    assert self_training._score_holdout(json_contract, '{"count":64,"name":"mlx"}') == 1.0
    assert self_training._score_holdout(json_contract, '{"name":"mlx","count":63}') == 0.0
    assert self_training._score_holdout(ordered, "alpha, beta, gamma") == 1.0
    assert self_training._score_holdout(ordered, "alpha, gamma, beta") == 0.0


class _TimedBackend:
    def __init__(self, measured: bool = True):
        self.measured = measured

    def generate_with_adapter_control(self, **kwargs):
        holder = kwargs["stats_holder"]
        holder["stats"] = {
            "timings": {"predicted_per_second": 20.0} if self.measured else {}
        }
        prompt = kwargs["messages"][0]["content"]
        if "17 * 23" in prompt:
            yield "391"
        elif "JSON object" in prompt:
            yield '{"name":"mlx","count":64}'
        elif "three tokens" in prompt:
            yield "alpha,beta,gamma"
        else:
            yield "revert and retrain"


def test_holdout_requires_real_backend_timing():
    score, speed, measured, receipts = self_training._run_holdout(_TimedBackend(), False)
    assert score == 1.0
    assert speed == 20.0
    assert measured is True
    assert len(receipts) == len(self_training._HOLDOUT_BENCHMARK)

    score, speed, measured, _ = self_training._run_holdout(_TimedBackend(measured=False), False)
    assert score == 1.0
    assert speed == 0.0
    assert measured is False


def test_reconcile_stale_training_unwedges_start_with_provenance(monkeypatch):
    state = self_training._empty_state()
    state.update({
        "status": "training",
        "lastJobId": "job-dead",
        "trainingQualifiedOnly": True,
    })
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "error"
    assert state["trainingQualifiedOnly"] is False
    assert state["lastRecovery"]["kind"] == "stale-training"
    assert state["lastRecovery"]["persistedJobId"] == "job-dead"


def test_reconcile_stale_manual_queue_never_replays_silently(monkeypatch):
    state = self_training._empty_state()
    state.update({"status": "queued", "trainingQualifiedOnly": False})
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "idle"
    assert state["lastRecovery"]["kind"] == "stale-queue-cleared"


def test_reconcile_stale_autonomous_queue_requires_current_policy(monkeypatch):
    state = self_training._empty_state()
    state.update({"status": "queued", "trainingQualifiedOnly": True})
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("", False))
    monkeypatch.setattr(self_training, "_autonomous_qlora_policy_allows", lambda current: True)

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is True
    assert state["status"] == "queued"
    assert state["trainingQualifiedOnly"] is True
    assert state["lastRecovery"]["kind"] == "stale-autonomous-queue-rescheduled"


def test_reconcile_unknown_runtime_state_is_fail_open(monkeypatch):
    state = self_training._empty_state()
    state.update({"status": "training", "lastJobId": "job-unknown"})
    before = dict(state)
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: (None, None))

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is False
    assert reschedule is False
    assert state == before
