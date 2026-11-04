"""Port of packages/evals CLI parsing plus docker orchestration helpers."""

import json

import pytest

from pi.evals.cli import (
    artifact_run_id,
    compare_discovery,
    container_path,
    normalize_discovered_file,
    parse_eval_cli,
)
from pi.evals.docker import docker_args, require_eval_auth_file, task_directory_name
from pi.evals.plan import EvalTask, parse_discovered_cases


def test_parses_requested_files_and_filters():
    options = parse_eval_cli(["evals/models.docs.eval.py", "-t", "adds the model"])
    assert options.requested_files == ["evals/models.docs.eval.py"]
    assert options.discovery_args == ["-t", "adds the model"]


def test_parses_test_name_pattern_prefix_form():
    options = parse_eval_cli(["--testNamePattern=adds"])
    assert options.discovery_args == ["--testNamePattern=adds"]


def test_parses_runner_options_in_both_forms():
    options = parse_eval_cli(["--provider=fixture", "--model", "model", "--runs-per-variant=3"])
    assert (options.provider, options.model, options.runs_per_variant) == ("fixture", "model", 3)


def test_missing_filter_value_is_rejected():
    with pytest.raises(ValueError, match="Missing value for -t"):
        parse_eval_cli(["-t"])


def test_empty_pattern_prefix_is_rejected():
    with pytest.raises(ValueError, match="Missing value for --testNamePattern"):
        parse_eval_cli(["--testNamePattern="])


def test_missing_runner_value_is_rejected():
    with pytest.raises(ValueError, match="Missing value for --provider"):
        parse_eval_cli(["--provider"])
    with pytest.raises(ValueError, match="Missing value for --model"):
        parse_eval_cli(["--model", "-t"])


def test_unsupported_arguments_are_rejected():
    with pytest.raises(ValueError, match="Unsupported eval argument"):
        parse_eval_cli(["--verbose"])


def test_cli_model_selection_requires_both_values():
    with pytest.raises(ValueError, match="requires both"):
        parse_eval_cli(["--provider", "fixture"], {})


def test_environment_defaults_and_pairing(monkeypatch):
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_MODEL", raising=False)
    with pytest.raises(ValueError, match="Set both PI_PROVIDER and PI_MODEL"):
        parse_eval_cli([], {"PI_PROVIDER": "fixture"})


def test_environment_defaults_are_used(monkeypatch):
    options = parse_eval_cli([], {"PI_PROVIDER": " fixture ", "PI_MODEL": "model"})
    assert (options.provider, options.model) == ("fixture", "model")


def test_runs_per_variant_defaults_and_environment(monkeypatch):
    assert parse_eval_cli([], {}).runs_per_variant == 1
    assert parse_eval_cli([], {"PI_EVAL_RUNS_PER_VARIANT": "4"}).runs_per_variant == 4
    with pytest.raises(ValueError, match="positive integer"):
        parse_eval_cli([], {"PI_EVAL_RUNS_PER_VARIANT": "0"})
    with pytest.raises(ValueError, match="positive integer"):
        parse_eval_cli([], {"PI_EVAL_RUNS_PER_VARIANT": "soon"})


def test_artifact_run_id_shape():
    run_id = artifact_run_id()
    stamp, _, identifier = run_id.rpartition("_")
    assert stamp.startswith("2") and "T" in stamp and identifier


def test_container_path_is_package_relative(tmp_path):
    from pi.evals.cli import package_root

    assert container_path("evals/examples.docs.eval.py") == "evals/examples.docs.eval.py"
    outside = str(package_root.parent / "outside.docs.eval.py")
    with pytest.raises(ValueError, match="Eval file must be inside"):
        container_path(outside)


def test_normalize_discovered_file_strips_container_prefix():
    assert normalize_discovered_file("/repo/pi-python/evals/models.docs.eval.py") == "evals/models.docs.eval.py"
    with pytest.raises(ValueError, match="outside the container package"):
        normalize_discovered_file("/repo/other/evals/models.docs.eval.py")


def test_compare_discovery_requires_identical_cohorts():
    container_file = "/repo/pi-python/evals/models.docs.eval.py"
    cases = parse_discovered_cases([{"name": "Add model > adds the model", "file": container_file}])
    other = parse_discovered_cases([{"name": "Add model > adds the model", "file": container_file}])
    compare_discovery(cases, other)
    different = parse_discovered_cases([{"name": "Add model > removes the model", "file": container_file}])
    with pytest.raises(ValueError, match="discovered different eval cases"):
        compare_discovery(cases, different)


def test_require_eval_auth_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path))
    auth = tmp_path / "auth.json"
    with pytest.raises(RuntimeError, match="does not exist"):
        require_eval_auth_file("fixture")
    auth.write_text(json.dumps({"fixture": {"type": "api_key", "key": "k"}}), encoding="utf-8")
    assert require_eval_auth_file("fixture") == str(auth.resolve())
    auth.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is invalid"):
        require_eval_auth_file("fixture")
    auth.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no credential for provider"):
        require_eval_auth_file("fixture")


def test_docker_args_mounts_and_environment(tmp_path):
    from pi.evals.docker import DockerContext

    context = DockerContext(
        images={"without_docs": {"name": "img-without:local", "id": "a"},
                "with_docs": {"name": "img-with:local", "id": "b"}},
        artifact_directory=str(tmp_path),
        auth_path="/host/auth.json",
        provider="fixture",
        model="model",
        runs_per_variant=2,
    )
    output_directory = str(tmp_path / "out")
    args = docker_args(context, "without_docs", output_directory, ["--discover", "evals/x.docs.eval.py"])
    assert args[:4] == ["run", "--rm", "--read-only", "--tmpfs"]
    joined = " ".join(args)
    assert "--env PI_EVAL_ARTIFACT_DIR=/artifacts" in joined
    assert "--env PI_EVAL_RUNS_PER_VARIANT=2" in joined
    assert "--env PI_EVAL_SANDBOX_UID=65532" in joined
    assert "--env PI_PROVIDER=fixture" in joined
    assert "--env PI_MODEL=model" in joined
    assert "type=bind,source=/host/auth.json,target=/run/pi-eval-secrets/auth.json,readonly" in joined
    assert "img-without:local" in args
    assert args[-2:] == ["--discover", "evals/x.docs.eval.py"]
    assert (tmp_path / "out").is_dir()


def test_task_directory_name_is_a_stable_digest():
    task = EvalTask(file="evals/x.docs.eval.py", full_name="Set > case", eval_set="Set", case_id="case",
                    variant="with_docs", model="fixture/model", run_number=1)
    assert task_directory_name(task) == task_directory_name(task)
    assert len(task_directory_name(task)) == 64
