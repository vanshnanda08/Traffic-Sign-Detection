"""Same-machine minimal reproduction checkout; copies declared inputs, never training stores."""
import argparse,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='.cache/reproduction_stage_v1');args=parser.parse_args()
    stage=ROOT/args.output;stage.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/reproduction_v1.json').read_text());m=json.loads((ROOT/cfg['manifest']).read_text());assets=json.loads((ROOT/'configs/reproduction_assets.json').read_text())
    def copy(relative):
        p=ROOT/relative;q=stage/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    shutil.copytree(ROOT/'src',stage/'src',ignore=shutil.ignore_patterns('__pycache__'))
    for path in ['run_all.sh','requirements.txt','configs/reproduction_v1.json','configs/evaluation_v1.json','configs/degradation_dev_v1.json','docs/reproduction_v1/release_manifest.json','docs/reproduction_v1/development_preflight.json','scripts/reproduce.py',*m['provenance_references']]:copy(path)
    core=Path(assets['core']);core=core if core.is_absolute() else ROOT/core
    for split,rows in m['splits'].items():
        for row in rows:
            for key in ['image_path','label_path']:
                target=stage/'data/benchmark'/row[key];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(core/row[key],target)
    local={'core':'data/benchmark','checkpoints':{}}
    for model,path in assets['checkpoints'].items():
        source=Path(path);source=source if source.is_absolute() else ROOT/source
        target=stage/'models'/f'{model}.pt';target.parent.mkdir(exist_ok=True);shutil.copy2(source,target);local['checkpoints'][model]=str(target.relative_to(stage))
    (stage/'configs/reproduction_assets.json').write_text(json.dumps(local,indent=2)+'\n')
    # Same installed scientific interpreter, honestly not a second computer or dependency reinstall.
    (stage/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True)
    files={str(p.relative_to(stage)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in stage.rglob('*') if p.is_file() and '.venv' not in p.parts}
    receipt={'stage':str(stage),'scope':'Same-machine minimal input checkout; shared installed .venv; no training runs/raw/reference/audit stores','inputs':'90val/test originals and labels;2canonical checkpoints; small code/config/provenance records','files':files,'total_file_bytes':sum(v['bytes'] for v in files.values())}
    (ROOT/'results/w8a_v1/minimal_stage.json').write_text(json.dumps(receipt,indent=2)+'\n');print(stage)
