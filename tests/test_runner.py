"""Release runner checks; original tests retained in scripts/historical."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_default_dispatch_is_genuine_evaluation_only():
    script=(ROOT/'run_all.sh').read_text()
    assert 'MODE="${1:---reproduce}"' in script and 'scripts/reproduce.py' in script
    assert 'train_main.py' not in script and 'train_recovery.py' not in script
    assert '--smoke' in script and 'historical_checks.py' in script
def test_reproduction_help_from_another_directory(tmp_path):
    run=subprocess.run(['bash',str(ROOT/'run_all.sh'),'--reproduce','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0 and '--device' in run.stdout and '--assets' in run.stdout
def test_unknown_mode_rejected():
    run=subprocess.run(['bash',str(ROOT/'run_all.sh'),'--typo'],capture_output=True,text=True)
    assert run.returncode==2 and 'Usage' in run.stderr
