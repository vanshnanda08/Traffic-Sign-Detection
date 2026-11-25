"""Freeze only after tests/development preflight, before condition generation."""
import hashlib,json,shutil
from pathlib import Path
from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.evaluation_data import config,CONFIG
from p57.evaluation_protocol import FILES,RECEIPT,library_sources,authenticate_models
from p57.durability import atomic_json

if __name__=='__main__':
    authenticate_models();cfg=config();pre=ROOT/'results/w6_verification_v1/preflight/result.json'
    if RECEIPT.exists():raise SystemExit('Evaluator already frozen; do not repin after test access')
    if (ROOT/cfg['derived_root']).exists() or (ROOT/cfg['output_root']/'runs').exists():raise SystemExit('Held-out output already exists before freeze')
    result=json.loads(pre.read_text())
    if result['status']!='passed' or result['config_sha256']!=file_sha256(CONFIG):raise SystemExit('Development preflight not passed for current config')
    for name in ['focused_tests','regressions']:
        if (ROOT/f'results/w6_verification_v1/{name}.exit').read_text().strip()!='0':raise SystemExit('Required tests not passed')
    files={name:file_sha256(ROOT/name) for name in FILES};snapshot=ROOT/'results/detector_eval_v1/source_snapshot'
    for name,wanted in files.items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target);assert file_sha256(target)==wanted
    value={'version':'evaluator_v1','config_sha256':file_sha256(CONFIG),'files':files,'snapshot':str(snapshot.relative_to(ROOT)),'library_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in library_sources().items()},'preflight_path':str(pre.relative_to(ROOT)),'preflight_sha256':file_sha256(pre),'checkpoint_hashes':cfg['checkpoints'],'git_commit':None,'tolerances_declared_before_preflight':cfg['tolerances'],'status':'frozen before any held-out condition generation or prediction'}
    atomic_json(RECEIPT,value);atomic_json(ROOT/'docs/evaluation_v1/installed_library_sources.json',library_sources());print('Frozen evaluator receipt SHA',file_sha256(RECEIPT))
