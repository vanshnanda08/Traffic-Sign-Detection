"""Read-only post-run W6 audit; never calls inference or changes frozen code."""
import os
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(TASK_ROOT / '.cache/ultralytics')
os.environ['MPLCONFIGDIR'] = str(TASK_ROOT / '.cache/matplotlib')
import csv
import json
import numpy as np
from p57.assets import file_sha256
from p57.durability import atomic_json
from p57.evaluation_data import config, verify_conditions, CONDITIONS
from p57.evaluation_protocol import verify_freeze, RECEIPT
from p57.evaluation_engine import recompute
from p57.evaluation_artifacts import display_matching, paired_table, METRICS


def read(path):
    return json.loads(path.read_text())


def main():
    cfg = config()
    receipt = verify_freeze()
    manifest = verify_conditions(deep=True)
    root = TASK_ROOT / cfg['output_root']
    out = root / 'artifacts'
    names = {int(v['id']): v['name'] for v in read(TASK_ROOT / cfg['core'] / 'dataset.lock.json')['class_mapping'].values()}
    tol = cfg['tolerances']['serialized_metric_absolute']
    progress = read(root / 'progress.json')
    assert progress['status'] == 'passed' and len(progress['completed']) == 14
    paired_manifest = {(r['condition'], r['source_id']): r for r in manifest['rows']}
    runs, audit, order, batches, inventory = {}, [], None, None, []
    canonical = read(out / 'metrics_14.json')
    assert len(canonical) == 14
    for idx, (condition, model) in enumerate((c, m) for c in CONDITIONS for m in ['A', 'C']):
        key = f'{model}_{condition}'
        run = root / 'runs' / key
        m = read(run / 'metrics.json')
        p = read(run / 'predictions.json')
        s = read(run / 'status.json')
        ex = read(root / 'supervision' / key / 'exit.json')
        assert s['status'] == 'passed' and ex['returncode'] == 0
        assert p['complete'] and p['scope'] == 'heldout' and p['condition'] == condition
        assert m['scope'] == 'heldout' and m['images'] == 45 and m['boxes'] == 45
        assert m['config_sha256'] == p['config_sha256'] == receipt['config_sha256']
        assert m['checkpoint_sha256'] == p['checkpoint_sha256'] == cfg['checkpoints'][model]['sha256']
        assert m['predictions_sha256'] == file_sha256(run / 'predictions.json')
        assert m['actual_device'].startswith('mps') and m['half'] is False
        for setting, value in cfg['settings'].items():
            assert m['settings'][setting] == value, setting
        assert m['settings']['data'] == str(TASK_ROOT / cfg['derived_root'] / condition / 'dataset.yaml')
        coverage = m['completion']
        assert coverage['complete'] and coverage['expected_images'] == coverage['nms_images'] == coverage['metric_images'] == 45
        assert len(coverage['batches']) == 12 and all(b['nms_completed'] for b in coverage['batches'])
        current_batches = [[Path(s).stem for s in b['source_ids']] for b in coverage['batches']]
        current_order = [r['source_id'] for r in p['records']]
        assert len(current_order) == len(set(current_order)) == 45 and current_order == m['source_order']
        if order is None:
            order, batches = current_order, current_batches
        assert current_order == order and current_batches == batches
        detail = recompute(p['records'], names, tol, detailed=True)
        assert all(abs(detail['aggregate'][k] - m['aggregate'][k]) <= tol for k in METRICS)
        assert len(detail['per_class']) == len(m['per_class']) == 6
        for a, b in zip(detail['per_class'], m['per_class']):
            assert a['id'] == b['id'] and all(abs(a[k] - b[k]) <= tol for k in METRICS)
        for r in p['records']:
            source = paired_manifest[(condition, r['source_id'])]
            assert r['condition'] == condition and r['image_sha256'] == source['derived_file_sha256']
            assert r['image_path'] == str(TASK_ROOT / cfg['derived_root'] / condition / source['image_path'])
            assert np.array_equal(np.asarray(r['classes']), np.asarray(r['matching']['stats']['pred_cls']))
            assert np.array_equal(np.asarray(r['confidence']), np.asarray(r['matching']['stats']['conf']))
            assert np.array_equal(np.asarray(r['ground_truth']['classes']), np.asarray(r['matching']['stats']['target_cls']))
            assert len(r['ground_truth']['classes']) == source['boxes']
            matched = display_matching(r)
            inventory.append({'source_id': r['source_id'], 'condition': condition, 'model': model,
                              **{k: len(matched[k]) for k in ['FP', 'FN', 'TP']}})
        c = canonical[idx]
        assert c['model'] == model and c['condition'] == condition and c['complete'] and c['boxes'] == 45
        assert c['per_class'] == m['per_class'] and all(c[k] == m['aggregate'][k] for k in METRICS)
        assert c['metrics_sha256'] == file_sha256(run / 'metrics.json')
        runs[(model, condition)] = m
        samples = [json.loads(line) for line in (root / 'supervision' / key / 'resources.jsonl').read_text().splitlines()]
        audit.append({'key': key, 'actual_exit': ex, 'metrics_sha256': file_sha256(run / 'metrics.json'),
                      'predictions_sha256': m['predictions_sha256'], 'input_shapes': sorted({tuple(r['input_shape']) for r in p['records']}),
                      'ground_truth_boxes': m['boxes'], 'resource_samples': len(samples),
                      'max_observed_rss_bytes': max(s['rss_bytes'] or 0 for s in samples),
                      'min_observed_available_bytes': min(s['available_bytes'] for s in samples),
                      'max_observed_global_swap_bytes': max(s['global_swap_bytes'] for s in samples)})
    paired = paired_table(runs)
    assert paired == read(out / 'paired_table.json')
    for name, expected in [('metrics_14.csv', canonical), ('paired_table.csv', paired)]:
        with (out / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(expected)
        for a, b in zip(rows, expected):
            for k, v in a.items():
                assert float(v) == b[k] if isinstance(b[k], (int, float)) and not isinstance(b[k], bool) else v == str(b[k])
    table = (out / 'main_table.md').read_text()
    for row in paired:
        expected = f"| {row['condition']} | {100*row['A_mAP50']:.3f} | {100*row['C_mAP50']:.3f} | {100*row['delta_mAP50_fraction']:+.3f} | {100*row['A_precision']:.3f} | {100*row['C_precision']:.3f} | {100*row['A_recall']:.3f} | {100*row['C_recall']:.3f} |"
        assert expected in table
    for family, plot in read(out / 'plot_data.json').items():
        assert plot['display_multiplier'] == 100 and plot['severity'][0] == 0 and plot['conditions'][0] == 'clean'
        for metric, models in plot['metrics_in_fraction_units'].items():
            for model, values in models.items():
                assert values == [runs[(model, c)]['aggregate'][metric] for c in plot['conditions']]
    failures = read(out / 'failure_inventory.json')
    key = lambda r: (r['condition'], r['source_id'], r['model'])
    assert sorted(inventory, key=key) == sorted(failures['inventory'], key=key) and len(inventory) == 630
    for row in failures['selected']:
        assert (root / row['file']).is_file()
        assert row['prediction_sources'] == {m: runs[(m, row['condition'])]['predictions_sha256'] for m in ['A', 'C']}
    for path, sha in read(out / 'artifact_manifest.json')['files'].items():
        assert file_sha256(out / path) == sha
    protected = read(TASK_ROOT / 'results/w6_verification_v1/protected_before.json')
    for path, before in protected.items():
        target = TASK_ROOT / path
        assert target.is_file() and file_sha256(target) == before['sha256'], path
        assert target.stat().st_mtime_ns == before['mtime_ns'], path
    frozen_time = RECEIPT.stat().st_mtime_ns
    assert frozen_time < min((TASK_ROOT / cfg['derived_root'] / r['condition'] / r['image_path']).stat().st_mtime_ns for r in manifest['rows'])
    assert frozen_time < min((root / 'runs' / r['key'] / 'predictions.json').stat().st_mtime_ns for r in audit)
    target = TASK_ROOT / 'docs/evaluation_v1/post_run_verification.json'
    assert not target.exists(), 'Preserve previous audit; use an explicit new receipt version'
    atomic_json(target, {'status': 'passed', 'scope': 'Read-only W6 post-run audit; no inference',
                        'evaluations': 14, 'model_source_evaluations': 630, 'paired_source_conditions': 315,
                        'boxes_per_condition': 45, 'historical_hashes_and_mtimes_unchanged': len(protected),
                        'exact_recomputed_aggregate_and_per_class': True, 'identical_source_and_batch_order': True,
                        'source_pixels_and_labels_reverified': True, 'table_and_plot_consistency': True,
                        'freeze_preceded_heldout_outputs': True, 'serialization_absolute_tolerance': tol,
                        'evaluator_receipt_sha256': file_sha256(RECEIPT), 'runs': audit})
    print(f'W6 post-run audit passed:14 evaluations/630 records; {len(protected)} historical files preserved')


if __name__ == '__main__':
    main()
