"""External assets (FST, CLAM, the private ArtifactNet tree, a local fma_full copy) are located by flags or
environment variables, never by a path that exists on one maintainer's machine."""
import argparse
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# A user-specific absolute location: /home/<name>/, /Users/<name>/, /media/<name>/, /Volumes/<name>/, /mnt/<name>/.
# Bare prefixes such as '/home/' (used by the release guards to reject private paths) do not match.
USER_PATH_RE = re.compile(r"/(?:home|Users|media|Volumes|mnt)/[A-Za-z0-9_.-]+/")


def test_no_user_specific_path_in_package_or_scripts():
    offenders = []
    for base in ("artifactbench", "scripts"):
        for path in sorted((ROOT / base).rglob("*.py")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if USER_PATH_RE.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "user-specific absolute paths:\n" + "\n".join(offenders)


def test_user_path_pattern_matches_maintainer_paths_but_not_guard_prefixes():
    assert USER_PATH_RE.search("root = Path('/home/someone/dev/ArtifactNet')")
    assert USER_PATH_RE.search("'/media/someone/Archive/dataset/fma_full'")
    assert not USER_PATH_RE.search("prefix in ('/home/', '/media/', '/Volumes/', '/Users/')")


# --- FST -----------------------------------------------------------------------------------------------------------

def test_fst_without_repo_raises_naming_flag_and_variable(monkeypatch):
    from artifactbench.models.fst import FSTModel
    monkeypatch.delenv("ARTIFACTBENCH_FST_REPO", raising=False)
    with pytest.raises(ValueError) as err:
        FSTModel()
    assert "--fst-repo" in str(err.value) and "ARTIFACTBENCH_FST_REPO" in str(err.value)


def test_fst_repo_from_environment(monkeypatch, tmp_path):
    from artifactbench.models.fst import FSTModel
    monkeypatch.setenv("ARTIFACTBENCH_FST_REPO", str(tmp_path))
    model = FSTModel()
    assert model.fst_repo == tmp_path
    assert model.stage1_ckpt == str(tmp_path / "checkpoints/backbone_stage1.ckpt")
    assert model.stage2_ckpt == str(tmp_path / "checkpoints/classifier_stage2.ckpt")


def test_fst_repo_argument_wins_over_environment(monkeypatch, tmp_path):
    from artifactbench.models.fst import FSTModel
    monkeypatch.setenv("ARTIFACTBENCH_FST_REPO", str(tmp_path / "env"))
    assert FSTModel(fst_repo=str(tmp_path / "arg")).fst_repo == tmp_path / "arg"


def test_bench_cli_wires_fst_flags_into_the_adapter(monkeypatch, tmp_path):
    from artifactbench.bench import build_model, build_parser
    from artifactbench.models.fst import FSTModel
    monkeypatch.delenv("ARTIFACTBENCH_FST_REPO", raising=False)
    args = build_parser().parse_args([
        "--model", "fst", "--manifest", "m.json", "--output", str(tmp_path / "out"),
        "--fst-repo", str(tmp_path / "fst"),
        "--fst-stage1-ckpt", str(tmp_path / "s1.ckpt"),
        "--fst-stage2-ckpt", str(tmp_path / "s2.ckpt"),
    ])
    model = build_model("fst", args)
    assert isinstance(model, FSTModel)
    assert model.fst_repo == tmp_path / "fst"
    assert model.stage1_ckpt == str(tmp_path / "s1.ckpt")
    assert model.stage2_ckpt == str(tmp_path / "s2.ckpt")


def test_bench_cli_fst_without_repo_fails_before_any_audio(monkeypatch, tmp_path):
    from artifactbench.bench import build_model, build_parser
    monkeypatch.delenv("ARTIFACTBENCH_FST_REPO", raising=False)
    args = build_parser().parse_args(["--model", "fst", "--manifest", "m.json", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="--fst-repo"):
        build_model("fst", args)


# --- pinned v1.2 assets ----------------------------------------------------------------------------------------------

def test_asset_location_without_variable_or_default_raises(monkeypatch):
    from artifactbench.v12.model_assets import asset_location
    monkeypatch.delenv("ARTIFACTBENCH_ARTIFACTNET_REPO", raising=False)
    with pytest.raises(ValueError, match="ARTIFACTBENCH_ARTIFACTNET_REPO"):
        asset_location("ARTIFACTBENCH_ARTIFACTNET_REPO")


def test_asset_location_variable_wins_over_default(monkeypatch, tmp_path):
    from artifactbench.v12.model_assets import asset_location
    monkeypatch.setenv("ARTIFACTBENCH_CLAM_REPO", str(tmp_path / "env"))
    assert asset_location("ARTIFACTBENCH_CLAM_REPO", str(tmp_path / "default")) == tmp_path / "env"


@pytest.mark.parametrize("name", ["artifactnet", "clam", "fst"])
def test_pinned_source_trees_need_a_location(monkeypatch, tmp_path, name):
    from artifactbench.v12.model_assets import build_pinned
    for variable in ("ARTIFACTBENCH_ARTIFACTNET_REPO", "ARTIFACTBENCH_CLAM_REPO", "ARTIFACTBENCH_FST_REPO"):
        monkeypatch.delenv(variable, raising=False)
    snapshots = tmp_path / "snapshots.json"
    snapshots.write_text("[]")
    with pytest.raises(ValueError, match="ARTIFACTBENCH_"):
        build_pinned(name, snapshots)


def test_freeze_exposure_audit_needs_the_artifactnet_tree(monkeypatch):
    from artifactbench.v12.freeze import local_exposure_audit
    monkeypatch.delenv("ARTIFACTBENCH_ARTIFACTNET_REPO", raising=False)
    with pytest.raises(ValueError, match="ARTIFACTBENCH_ARTIFACTNET_REPO"):
        local_exposure_audit([], {"input_hashes": {}})


# --- maintainer re-run tool ------------------------------------------------------------------------------------------

def test_rerun_current_requires_the_artifactnet_tree(monkeypatch, tmp_path, capsys):
    from artifactbench import rerun_current
    monkeypatch.delenv("ARTIFACTBENCH_ARTIFACTNET_REPO", raising=False)
    with pytest.raises(SystemExit):
        rerun_current.parse_args(["--output", str(tmp_path)])
    message = capsys.readouterr().err
    assert "--artifactnet-repo" in message and "ARTIFACTBENCH_ARTIFACTNET_REPO" in message


def test_rerun_current_fma_recovery_requires_a_root(monkeypatch, tmp_path, capsys):
    from artifactbench import rerun_current
    monkeypatch.setenv("ARTIFACTBENCH_ARTIFACTNET_REPO", str(tmp_path / "an"))
    monkeypatch.delenv("ARTIFACTBENCH_FMA_FULL_ROOT", raising=False)
    with pytest.raises(SystemExit):
        rerun_current.parse_args(["--output", str(tmp_path), "--recover-fma-from", str(tmp_path / "prev")])
    message = capsys.readouterr().err
    assert "--fma-full-root" in message and "ARTIFACTBENCH_FMA_FULL_ROOT" in message


def test_rerun_current_reads_locations_from_flags_and_environment(monkeypatch, tmp_path):
    from artifactbench import rerun_current
    monkeypatch.setenv("ARTIFACTBENCH_ARTIFACTNET_REPO", str(tmp_path / "an"))
    monkeypatch.setenv("ARTIFACTBENCH_FMA_FULL_ROOT", str(tmp_path / "fma"))
    args = rerun_current.parse_args(["--output", str(tmp_path), "--recover-fma-from", str(tmp_path / "prev")])
    assert args.artifactnet_repo == tmp_path / "an" and args.fma_full_root == tmp_path / "fma"
    args = rerun_current.parse_args(["--output", str(tmp_path), "--artifactnet-repo", str(tmp_path / "flag")])
    assert args.artifactnet_repo == tmp_path / "flag"
    assert isinstance(args, argparse.Namespace)
