# SPDX-License-Identifier: AGPL-3.0-only

import asyncio

from fastapi import BackgroundTasks

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
    monkeypatch.setattr(self_training, "_training_terminal_snapshot", lambda _job=None: None)

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


def test_gguf_repo_maps_to_trainable_source_without_guessing_arbitrary_local_file(tmp_path):
    assert (
        self_training._trainable_baseline_for_serving_model("unsloth/Qwen3.8-27B-GGUF")
        == "unsloth/Qwen3.8-27B"
    )
    standalone = tmp_path / "custom.gguf"
    standalone.write_bytes(b"not-a-real-gguf")
    assert self_training._trainable_baseline_for_serving_model(str(standalone)) == str(standalone)


def test_hf_cache_gguf_path_maps_to_trainable_source(tmp_path):
    model_root = tmp_path / "hub" / "models--unsloth--Qwen3.8-27B-GGUF"
    blob = model_root / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"not-a-real-gguf")
    target = model_root / "snapshots" / "abc" / "Qwen3.8-27B-Q4.gguf"
    target.parent.mkdir(parents=True)
    target.symlink_to(blob)
    assert (
        self_training._trainable_baseline_for_serving_model(str(target))
        == "unsloth/Qwen3.8-27B"
    )


def test_collected_gguf_turn_separates_serving_and_trainable_identity(monkeypatch):
    state = self_training._empty_state()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = asyncio.run(
        self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                modelId="unsloth/Qwen3.8-27B-GGUF",
                prompt="task",
                completion="answer",
            ),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )

    assert result["recorded"] is True
    assert state["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert state["baseServingModelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert state["examples"][0]["modelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert state["examples"][0]["trainableBaseModelId"] == "unsloth/Qwen3.8-27B"


def test_explicit_trainable_baseline_keeps_matching_gguf_examples(monkeypatch):
    state = self_training._empty_state()
    state["examples"] = [
        {
            "modelId": "unsloth/Qwen3.8-27B-GGUF",
            "trainableBaseModelId": "unsloth/Qwen3.8-27B",
            "prompt": "task",
            "completion": "answer",
        },
        {
            "modelId": "other/Model-GGUF",
            "trainableBaseModelId": "other/Model",
            "prompt": "wrong",
            "completion": "wrong",
        },
    ]
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = self_training.set_self_training_baseline(
        self_training.BaselineRequest(modelId="unsloth/Qwen3.8-27B"),
        current_subject="unsloth",
    )

    assert result["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert len(state["examples"]) == 1
    assert state["examples"][0]["modelId"] == "unsloth/Qwen3.8-27B-GGUF"


def test_different_serving_family_is_not_mixed_into_existing_baseline(monkeypatch):
    state = self_training._empty_state()
    state["baseModelId"] = "unsloth/Qwen3.8-27B"
    state["baseServingModelId"] = "unsloth/Qwen3.8-27B-GGUF"
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = asyncio.run(
        self_training.record_self_training_example(
            self_training.SelfTrainingExampleRequest(
                modelId="other/Model-GGUF",
                prompt="task",
                completion="answer",
            ),
            BackgroundTasks(),
            current_subject="unsloth",
        )
    )
    assert result["recorded"] is False
    assert result["reason"] == "model differs from baseline"


def test_legacy_gguf_state_migrates_to_trainable_base_and_preserves_serving_snapshot():
    state = self_training._empty_state()
    state.update(
        {
            "baseModelId": "unsloth/Qwen3.8-27B-GGUF",
            "baseSnapshotPath": "/cache/qwen3.8-gguf/snapshot",
            "examples": [
                {
                    "modelId": "unsloth/Qwen3.8-27B-GGUF",
                    "prompt": "task",
                    "completion": "answer",
                }
            ],
        }
    )
    assert self_training._migrate_trainable_baseline(state) is True
    assert state["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert state["baseServingModelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert state["baseSnapshotPath"] is None
    assert state["baseServingSnapshotPath"] == "/cache/qwen3.8-gguf/snapshot"
    assert state["examples"][0]["trainableBaseModelId"] == "unsloth/Qwen3.8-27B"
    assert self_training._migrate_trainable_baseline(state) is False


def test_setting_gguf_baseline_never_marks_gguf_snapshot_as_trainable(monkeypatch):
    state = self_training._empty_state()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda _state: None)
    monkeypatch.setattr(self_training, "_write_dataset", lambda _state, **_kwargs: None)

    result = self_training.set_self_training_baseline(
        self_training.BaselineRequest(
            modelId="unsloth/Qwen3.8-27B-GGUF",
            snapshotPath="/cache/gguf-snapshot",
        ),
        current_subject="unsloth",
    )
    assert result["baseModelId"] == "unsloth/Qwen3.8-27B"
    assert result["baseServingModelId"] == "unsloth/Qwen3.8-27B-GGUF"
    assert result["baseSnapshotPath"] is None
    assert result["baseServingSnapshotPath"] == "/cache/gguf-snapshot"


def test_reconcile_completed_training_becomes_candidate_ready(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "training",
        "lastJobId": "job-done",
        "trainingQualifiedOnly": True,
    })
    monkeypatch.setattr(self_training, "_training_runtime_snapshot", lambda: ("job-done", False))
    monkeypatch.setattr(
        self_training,
        "_training_terminal_snapshot",
        lambda _job=None: {
            "job_id": "job-done",
            "active": False,
            "completed": True,
            "error": None,
            "output_dir": str(adapter),
            "message": "Training completed",
        },
    )

    changed, reschedule = self_training._reconcile_persisted_training_state(state)

    assert changed is True
    assert reschedule is False
    assert state["status"] == "candidate-ready"
    assert state["lastCandidateAdapterPath"] == str(adapter.resolve())
    assert state["candidateQualifiedOnly"] is True
    assert state["trainingQualifiedOnly"] is False
    assert state["lastRecovery"]["kind"] == "training-completed"


def test_completed_candidate_autobenchmark_requires_current_autonomous_policy(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "candidate-ready",
        "candidateQualifiedOnly": True,
        "lastCandidateAdapterPath": str(adapter),
    })
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: False)
    assert self_training._queue_candidate_benchmark_if_allowed(state) is None
    assert state["status"] == "candidate-ready"

    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: True)
    assert self_training._queue_candidate_benchmark_if_allowed(state) == str(adapter.resolve())
    assert state["status"] == "benchmark-queued"


def test_live_training_watcher_runs_autonomous_benchmark_after_valid_artifact(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "training",
        "lastJobId": "job-live",
        "trainingQualifiedOnly": True,
    })
    writes = []
    benchmarks = []
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda current: writes.append(dict(current)))
    monkeypatch.setattr(
        self_training,
        "_training_terminal_snapshot",
        lambda _job=None: {
            "job_id": "job-live",
            "active": False,
            "completed": True,
            "error": None,
            "output_dir": str(adapter),
            "message": "Training completed",
        },
    )
    monkeypatch.setattr(self_training, "_autonomous_candidate_benchmark_allowed", lambda _state: True)

    async def _no_sleep(_seconds):
        return None

    async def _benchmark(subject, path):
        benchmarks.append((subject, path))

    monkeypatch.setattr(self_training.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(self_training, "_benchmark_completed_candidate", _benchmark)

    asyncio.run(self_training._watch_self_training_job("unsloth", "job-live", True))

    assert state["status"] == "benchmark-queued"
    assert state["lastCandidateAdapterPath"] == str(adapter.resolve())
    assert benchmarks == [("unsloth", str(adapter.resolve()))]
    assert writes


def test_interrupted_benchmark_recovers_to_policy_gated_candidate_boundary(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({
        "status": "benchmarking",
        "candidateQualifiedOnly": True,
        "lastCandidateAdapterPath": str(adapter),
    })
    changed, reschedule = self_training._reconcile_persisted_training_state(state)
    assert changed is True
    assert reschedule is False
    assert state["status"] == "candidate-ready"
    assert state["lastRecovery"]["kind"] == "interrupted-benchmark-recovered"


def test_candidate_recovery_accepts_standard_peft_artifact_name(tmp_path):
    adapter = tmp_path / "peft-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    assert self_training._candidate_artifact_path(str(adapter)) == str(adapter.resolve())


def test_objective_benchmark_promotes_only_after_score_and_speed_pass(tmp_path, monkeypatch):
    import core.inference as inference_core

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapters.safetensors").write_bytes(b"fixture")
    state = self_training._empty_state()
    state.update({"baseModelId": "base/model", "status": "candidate-ready"})
    writes = []

    class _Backend:
        def __init__(self):
            self.active = False
            self.hot_swaps = 0
            self.reverts = 0

        def hot_swap_adapter(self, path, name, base_model):
            assert path == str(adapter.resolve())
            assert base_model == "base/model"
            self.hot_swaps += 1
            self.active = True
            return True

        def revert_to_base_model(self, base_model):
            assert base_model == "base/model"
            self.reverts += 1
            self.active = False
            return True

        def generate_with_adapter_control(self, **kwargs):
            holder = kwargs["stats_holder"]
            using_candidate = bool(kwargs.get("use_adapter"))
            holder["stats"] = {
                "timings": {"predicted_per_second": 30.0 if using_candidate else 20.0}
            }
            prompt = kwargs["messages"][0]["content"]
            if using_candidate:
                if "17 * 23" in prompt:
                    yield "391"
                elif "JSON object" in prompt:
                    yield '{"name":"mlx","count":64}'
                elif "three tokens" in prompt:
                    yield "alpha,beta,gamma"
                else:
                    yield "revert and retrain"
            else:
                yield "wrong"

    backend = _Backend()
    monkeypatch.setattr(self_training, "_read_state", lambda: state)
    monkeypatch.setattr(self_training, "_write_state", lambda current: writes.append(dict(current)))
    monkeypatch.setattr(inference_core, "get_inference_backend", lambda: backend)

    result = asyncio.run(
        self_training.benchmark_self_training_candidate(
            self_training.SelfTrainingBenchmarkRequest(
                candidateAdapterPath=str(adapter), adapterName="candidate"
            ),
            current_subject="unsloth",
        )
    )
    ev = result["lastEvaluation"]
    assert ev["baseScore"] == 0.0
    assert ev["candidateScore"] == 1.0
    assert ev["baseTokPerSec"] == 20.0
    assert ev["candidateTokPerSec"] == 30.0
    assert ev["speedMeasured"] is True
    assert ev["promoted"] is True
    # Candidate attach for measurement, revert for base measurement, then attach
    # again only after the objective gates passed.
    assert backend.hot_swaps == 2
    assert backend.reverts == 1
    assert backend.active is True
    assert result["status"] == "promoted"
    assert result["activeAdapterPath"] == str(adapter.resolve())
    assert writes
