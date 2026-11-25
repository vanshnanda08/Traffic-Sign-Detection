import subprocess

from p57.environment import ROOT


def test_full_mode_reports_incomplete_without_experiment_artifacts():
    before = set(ROOT.rglob("*.pt"))
    run = subprocess.run([str(ROOT / "run_all.sh"), "--full"], capture_output=True, text=True, timeout=120)
    assert run.returncode == 2, run.stdout + run.stderr
    assert "INCOMPLETE" in run.stderr
    assert "No experiment metrics" in run.stderr
    assert set(ROOT.rglob("*.pt")) == before


def test_runner_is_invocable_from_another_directory(tmp_path):
    run = subprocess.run([str(ROOT / "run_all.sh"), "--validate"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr


def test_unknown_mode_rejected():
    run = subprocess.run([str(ROOT / "run_all.sh"), "--typo"], capture_output=True, text=True)
    assert run.returncode == 2
    assert "Usage" in run.stderr
