"""Create a new immutable W5 source overlay before either main run."""
import json
import shutil
from pathlib import Path
from p57.assets import file_sha256
from p57.dataset import ROOT, save_json
from p57.protocol import validate_freeze

FILES=['src/p57/main_training.py','src/p57/main_verification.py','scripts/train_main.py',
       'scripts/verify_main_pair.py','scripts/check_main_run.py','scripts/create_main_overlay.py','tests/test_main_training.py']
if __name__=='__main__':
    _,receipt=validate_freeze()
    output=ROOT/'docs/main_ac_v1/execution_overlay.json'
    if output.exists():raise SystemExit('Overlay exists; preserve and version it instead of replacing')
    snapshot=ROOT/'results/main_ac_v1/execution_snapshot';snapshot.mkdir(parents=True,exist_ok=False)
    files={name:file_sha256(ROOT/name) for name in FILES}
    for name,sha in files.items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        assert file_sha256(target)==sha
    save_json(output,{'version':'W5-execution-v1','scope':'Identical additive execution code for A/C; no W4 scientific component changed',
          'protocol_sha256':receipt['protocol_sha256'],'W4_receipt_sha256':file_sha256(ROOT/'docs/protocol_w4_v1/freeze_receipt.json'),
          'snapshot':str(snapshot.relative_to(ROOT)),'files':files,'git_commit':None})
    print('Created overlay SHA256',file_sha256(output))
