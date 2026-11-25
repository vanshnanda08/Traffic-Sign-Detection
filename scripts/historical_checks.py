"""Unchanged legacy checker in a temporary materialized historical interface view.
Original receipts are never repinned; root release runner is versioned separately.
Only read-only modes are allowed. Historical source/data/checkpoints are unchanged.
"""
import os
from pathlib import Path
import shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in {'--validate','--smoke','--dataset-validate','--protocol'}:raise SystemExit('Only read-only historical checks allowed')
    with tempfile.TemporaryDirectory(prefix='p57-historical-checks-',dir=ROOT/'.cache') as temp:
        stage=Path(temp)
        for name in ['src','tests','configs','docs','app','scripts']:
            shutil.copytree(ROOT/name,stage/name,ignore=shutil.ignore_patterns('__pycache__'))
        for name in ['requirements.txt','pyproject.toml','.gitignore']:shutil.copy2(ROOT/name,stage/name)
        shutil.copy2(ROOT/'scripts/historical/run_all_pre_w8a.sh',stage/'run_all.sh')
        shutil.copy2(ROOT/'scripts/historical/test_runner_pre_w8a.py',stage/'tests/test_runner.py')
        (stage/'tests/test_reproduction.py').unlink(missing_ok=True)
        # Development authentication deliberately rejects linked ancestors. Materialize
        # its small core/view copies; keep unrelated raw/reference stores read-only links.
        (stage/'data/processed').mkdir(parents=True)
        for entry in (ROOT/'data').iterdir():
            if entry.name!='processed':(stage/'data'/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
        for entry in (ROOT/'data/processed').iterdir():
            target=stage/'data/processed'/entry.name
            if entry.name in {'p57_core300_v2','p57_development_v1'}:shutil.copytree(entry,target)
            else:target.symlink_to(entry,target_is_directory=entry.is_dir())
        for name in ['results','checkpoints','.venv']:(stage/name).symlink_to(ROOT/name,target_is_directory=True)
        env={**os.environ,'PYTHONPATH':str(stage/'src')}
        env.setdefault('P57_RUN_DIR',str(ROOT/'results/legacy_checks'/('check_'+str(os.getpid()))))
        print('Historical read-only interface stage:',stage,flush=True)
        raise SystemExit(subprocess.run(['bash',str(stage/'run_all.sh'),*sys.argv[1:]],cwd=stage,env=env).returncode)
