"""Explicit exact-assets packaging/install/verification. Never downloads or runs inference."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p57.release_assets import build,install,safe_extract,expected_bundle,authority,sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','extract','install','verify'])
 p.add_argument('--archive');p.add_argument('--output');p.add_argument('--root',default=str(ROOT))
 args=p.parse_args();root=Path(args.root)
 if args.mode=='build':
  if not args.output:p.error('build requires a new --output directory')
  report=build(root,args.output)
 elif args.mode in ['extract','install']:
  if not args.archive:p.error('archive required')
  receipt=json.loads((root/'docs/release_v1/assets_receipt.json').read_text())
  inventory=json.loads((root/'docs/release_v1/asset_inventory.json').read_text())
  if args.mode=='install':report=install(root,args.archive,receipt,inventory)
  else:
   if not args.output:p.error('extract requires a new --output directory')
   if sha(args.archive)!=receipt['archive_sha256']:raise ValueError('Archive SHA256 mismatch')
   entries,_=expected_bundle(root);expected={**entries,'INVENTORY.json':{'sha256':receipt['inventory_sha256']}}
   report=safe_extract(args.archive,args.output,expected)
 else:
  if sys.version_info[:3]!=(3,12,16):raise RuntimeError('Use pinned Python 3.12.16')
  if sys.platform!='darwin':raise RuntimeError('This release setup supports macOS only; other systems are unverified')
  # Importing the runner sets its no-download/cache defaults, but never calls main.
  import importlib.util
  spec=importlib.util.spec_from_file_location('p57_release_runner',root/'scripts/reproduce.py')
  runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
  cfg,m,core,models,versions=runner.authenticate(root)
  from p57.evaluation_engine import require_local_font
  font=require_local_font()
  import platform,fcntl
  report={'status':'passed','scope':'asset/software/font verification only; no inference','python':platform.python_version(),
          'platform':platform.platform(),'architecture':platform.machine(),'versions':versions,'font':font,
          'model_hashes':{key:sha(path) for key,path in models.items()},'core':str(core),'plain_runner_ready':True}
 print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
