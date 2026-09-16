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
