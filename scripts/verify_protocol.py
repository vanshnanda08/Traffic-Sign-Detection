"""Verify the frozen W4 detector protocol; never train or evaluate a model."""
import argparse
import json
from pathlib import Path
from p57.protocol import validate_freeze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-dir',type=Path,required=True);args=p.parse_args()
    protocol,receipt=validate_freeze()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    result={'status':'passed','scope':'W4 detector freeze verification only; no training/inference',
            'protocol_sha256':receipt['protocol_sha256'],'expected_budget':protocol['budget_summary']}
    (args.report_dir/'protocol_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
