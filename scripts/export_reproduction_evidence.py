"""Create reviewable genuine small evidence copies; never fabricates a transcript."""
import argparse,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    status=json.loads((run/'status.json').read_text())
    if status['status']!='passed' or status['fresh_evaluations']!=14:raise SystemExit('No release evidence from incomplete/failed reproduction')
    out=ROOT/'release_evidence/w8a_v1';(out/'logs').mkdir(parents=True,exist_ok=True)
    stage=run.parents[2];copies={}
    def copy(source,target):
        dest=out/target;dest.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes()
        if source.suffix in {'.json','.log','.md','.csv','.txt','.xml'}:
            data=raw.decode().replace(str(stage),'$REPRO_CHECKOUT').replace(str(ROOT),'$PROJECT_ROOT').encode()
            redaction='Only exact staging/project root path-prefix substitution; no numeric/log event edits'
        else:data=raw;redaction='none; byte-identical'
        if dest.exists():raise FileExistsError('Evidence exists; preserve/version rather than replace: '+str(dest))
        dest.write_bytes(data);copies[target]={'raw_source':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),'raw_sha256':sha(source),'portable_sha256':sha(dest),'redaction':redaction,'bytes':len(data)}
    for name in ['status.json','reproduction_freeze.json','generation_receipt.json','progress.json']:copy(run/name,name)
    for source in (run/'artifacts').iterdir():
        if source.is_file():copy(source,'artifacts/'+source.name)
    for directory in sorted((run/'logs').iterdir()):
        key=directory.name
        for name in ['stdout.log','stderr.log']:copy(directory/name,'logs/'+key+'_'+name)
        for name in ['command.json','exit.json']:copy(directory/name,'processes/'+key+'_'+name)
        copy(run/'runs'/key/'metrics.json','metrics/'+key+'.json')
    for name in ['development_preflight_contract.log','focused_release.log','historical_regressions.log','historical_regressions_materialized.log','reproduction.log']:
        copy(ROOT/'results/w8a_v1'/name,'logs/'+name)
    for path in ['docs/reproduction_v1/development_preflight.json','docs/dataset_v2/validation.json','docs/evaluation_v1/measured/metrics_14.json','docs/evaluation_v1/measured/paired_table.json']:
        copy(ROOT/path,'references/'+Path(path).name.replace('.json','_'+Path(path).parent.name+'.json'))
    rows=json.loads((run/'artifacts/metrics_14.json').read_text());ledger=[]
    for r in rows:
        key=r['model']+'_'+r['condition']
        for metric in ['mAP50','precision','recall','mAP50_95']:
            ledger.append({'claim':key+'/'+metric,'value_fraction':r[metric],'source':'metrics/'+key+'.json','json_field':'aggregate.'+metric,'executed_log':'logs/'+key+'_stdout.log','stderr':'logs/'+key+'_stderr.log','original_predictions_sha256':r['prediction_sha256'],'display_rule':'percentage=100*fraction; mAP delta pp=100*(C-A), no manual replacement'})
        ledger.append({'claim':key+'/runtime_including_cleanup_seconds','value':r['elapsed_including_cleanup'],'source':'processes/'+key+'_exit.json','json_field':'elapsed_seconds_including_cleanup'})
    ledger.extend([{'claim':'reproduction_cardinality','sources_per_condition':45,'conditions':7,'models':2,'model_source_evaluations':630,'source':'status.json','derivation':'14actual exit0 evaluations x45sources; generation receipt315pairs=7x45'},
                   {'claim':'core_dataset_counts','images':300,'boxes':301,'split_images':[210,45,45],'source':'references/validation_dataset_v2.json','provenance':'Historical executed dataset validation; not a new full-source audit in W8A'},
                   {'claim':'reproduction_total_runtime_seconds','value':status['elapsed_seconds_including_children_cleanup'],'source':'status.json','json_field':'elapsed_seconds_including_children_cleanup','scope':'generation+all child startup/evaluation/cleanup+artifact generation; not training runtime'}])
    (out/'number_to_log_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (out/'copy_provenance.json').write_text(json.dumps({'method':'Copies of genuinely executed files; original files retained. Prefix redactions only. Not a reconstructed execution transcript.','files':copies},indent=2)+'\n')
    (out/'README.md').write_text('# W8A genuine local release evidence\n\nFresh CPU inference14runs/630sources in a same-machine minimal staging checkout.\nNo old prediction replay or table copying. Original full logs/predictions remain\nignored in the recorded staging run. Small reviewed text copies sanitize only\nlocal root path prefixes, with raw/portable hashes in copy_provenance.json.\n\nnumber_to_log_ledger.json maps full-precision metrics, counts and runtime to\nactual metrics/logs/exits. Table percentages=100*fraction; differences pp=100*(C-A).\nPredictions remain ignored; their exactSHAis recorded for traceability.\nNo public commit/release or second-machine check is claimed.\n')
    print('Prepared',len(copies),'genuine evidence files,',sum(x['bytes'] for x in copies.values()),'bytes')
