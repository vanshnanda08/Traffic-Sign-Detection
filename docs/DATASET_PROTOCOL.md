# Phase 1 dataset protocol

## Active revision and preservation

The repair revision is **p57_core300_v2**, configured by
`configs/dataset_v2.json`. `configs/dataset_active.json` explicitly selects it
for commands without a config flag. The historical `configs/dataset.json`,
`data/processed/p57_core300/`, `results/dataset/`, `docs/dataset/` and original
`phase1_verification.md` are retained unchanged. They document earlier coverage,
not current clearance: independent review found accepted cross-split geometric
relationships missed by the approximate full-source search.

Corrected output: `data/processed/p57_core300_v2/`; corrected audit:
`results/dataset_v2/`; versionable evidence: `docs/dataset_v2/`.
See [repair verification](phase1_repair_verification.md) for executed outcomes.
The historical figures and selection table below are explicitly v1 records.

## Corrected relationship protocol (v2)

Numerical rules are unchanged: 256px SIFT, up to 180 features, ratio <0.7,
RANSAC reprojection threshold 3.0, at least 10 inliers, fraction ≥0.65, coverage
≥0.06, plus the existing outside-box rule for cross-class pairs. Filename,
exact-byte/decoded-pixel and exhaustive pHash≤6/dHash≤10 checks remain. Geometry
checks both directed matches in canonical path order and accepts a pair if either
direction meets the same rules. Reset OpenCV RNG to seed **42 before every
RANSAC call**, making results independent of earlier pairs and RNG consumption.
Apply the same pair function to full-source candidates and selected-core checks.

Full-source FLANN retrieval remains an optimization, not a certificate of absence.
For **every selected core**, check **all 44,850 unordered pairs** (both directions)
without FLANN/candidate pruning or reliance on stored group IDs. Necessary rule
failures (for example too few outside-box query keypoints) may reject a direction
without unnecessary matching; every pair is still visited. Save source paths/IDs,
assigned splits, per-direction match/inlier/coverage measurements, reasons and
dispositions for all accepted relationships. No relationship waivers were applied.

Union accepted findings from the original exhaustive check into the **full-source
component graph before selection**, then preserve source-test quarantine. Audit
each new candidate core exhaustively; merge additional relationships into full
components, reselect from seed 42, and repeat. Select at most one image per
component, so even accepted within-split relationships trigger reselection.
If class/count support fails or closure is not reached, record a blocker and do
not freeze a benchmark. Never repair a pair by moving one isolated image.

Preparation checks closure before writing a new benchmark. Frozen validation
independently reruns exhaustive checks on **actual prepared image bytes**, after
hash/annotation/mapping/cardinality checks. A stored group assignment or old passing
JSON cannot bypass the geometric check. Unresolved accepted pairs produce a failed
report and nonzero exit, including when the old group IDs are unique.

The original core has 18 accepted relationships, seven crossing splits, under
the deterministic bidirectional policy. Both reviewed pairs remain accepted:
70093/70023 and 65015/65017. Their original measurements, complete component
dispositions and revised selection are retained in `docs/dataset_v2/`.
The first v2 candidate had two additional within-training relationships; the
next candidate had zero. All 300 final images and 301 original boxes are retained
without image/geometry changes; only class-ID tokens are remapped.

| New ID | Source ID | Class | Train | Val | Test | Images |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | COMPULSARY_KEEP_RIGHT | 37 | 8 | 8 | 53 |
| 1 | 42 | NO_ENTRY | 37 | 8 | 8 | 53 |
| 2 | 44 | NO_STOPPING_OR_STANDING | 37 | 8 | 8 | 53 |
| 3 | 57 | ROUNDABOUT | 25 | 7 | 7 | 39 |
| 4 | 64 | SPEED_LIMIT_30 | 37 | 7 | 7 | 51 |
| 5 | 70 | SPEED_LIMIT_80 | 37 | 7 | 7 | 51 |
| | | **Total** | **210** | **45** | **45** | **300** |

The same existing class-selection rule chose ROUNDABOUT: pedestrian crossing
now has 26 eligible development groups, below the unchanged minimum 30; roundabout
has 32. Do not assume one box per image: one selected image has two valid boxes.
All annotations are preserved. Roundabout mixes warning/mandatory designs;
GTSRB class 40 is **not** an unconditional counterpart for this source class.
Compatible mappings and actual GTSRB overlap remain future reference gates.

Corrected frozen lock SHA-256:
`ec2b3d0acef8bc03717a606313aae6b6b10ad91d37ebb6e6eecdda72c1fded51`.

### Exact versioned commands

```bash
bash run_all.sh --smoke --config configs/dataset_v2.json
bash run_all.sh --dataset --config configs/dataset_v2.json
bash run_all.sh --dataset-validate --config configs/dataset_v2.json
```

Existing mode-only commands remain supported, selecting the explicit active-config
pointer. CLI `--config` overrides `P57_DATASET_CONFIG`, which overrides that pointer;
direct Python dataset commands accept `--config` too. Runner output prints the
revision/output path. `P57_RUN_DIR` can place fresh run evidence separately; by
default reports go under `results/dataset_v2/runs/<mode>/`, not historical logs.
Frozen validation supports a separate `--report-dir`. To demonstrate rejection
of the historical freeze without overwriting it:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py validate \
  --config configs/dataset.json --report-dir results/phase1_repair/legacy_rejection
```

That historical validation is expected to exit nonzero. Audit/preparation refuse
to write the historical config's output/audit paths. Corrected graph/evidence can
be recomputed by the versioned full runner; changed prepared files are never
silently repaired. The original freeze/config/historical evidence have a before/
after SHA-256 and mtime inventory; new verification is a separate record.

Device CLI precedence is explicit `--device`, then `configs/environment.json`
when the flag is absent. The default config is unchanged. Unavailable explicit
MPS still raises a clear error; CPU is not silently overridden by `auto`.

Limits are unchanged: sign crops/illustrations dominate, geographic/track origins
are unknown, general Indian-road performance is unverified, and GTSRB overlap is
unmeasured. Reports say **“No relationships detected under this protocol”**, never
claim physical-sign independence. Matching alone does not prove sign identity.

## Changes from the Proposal

The proposal requested 300 self-collected/self-annotated Indian street images.
Faculty has approved using public online data instead, **as reported by the
student on 2026-11-20**. We reuse public images/annotations and do not claim
ownership of their collection or annotation. The source mixes road scenes,
sign crops and illustrations; that is a substantive domain limitation, not a
dataset of 300 verified Indian street photographs.

## Source, version, license and attribution

- Creator/uploader: **SDI**, project **Indian Traffic Sign**, Roboflow Universe.
- [Version 1](https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1),
  generated June 8, 2023; supplied export README dated November 25, 2023.
- Format: YOLOv8 object detection, normalized `class xc yc width height` rows.
- Source-declared license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Attribute SDI/Roboflow Universe, retain the source/license link, and describe
  subset selection, class-ID remapping and new development split as changes.
- Archive: `data/raw/Indian Traffic Sign.v1i.yolov8.zip`.
- Archive SHA-256:
  `d4ce92dfd534681f4f0a70de3a217a4a79e7146382450c3c030888244306a762`.
- Source preprocessing: EXIF auto-orientation, stripping orientation metadata,
  640×640 **stretch** resize. These exported images are not camera originals.
- Source README and version page explicitly report **no Roboflow augmentations**.
  This does not establish the absence of transformations/duplicates upstream.

The public overview now also lists version 2. Do not substitute it for version 1.
The source has no collection/geographic/track-provenance description sufficient
to verify that every image originated in India or is independent of GTSRB.
Attribution identifies the dataset distributor; upstream provenance is unresolved.

## Historical v1 full-source audit (superseded relationship coverage)

All **6,750 images**, **6,750 labels**, **6,769 boxes** and **85 classes** were
analyzed. Source partitions contain 4,690 train, 1,383 validation and 677 test
images. There are zero decoding/label-format/box-bound errors, missing labels,
orphan labels, empty labels or conflicting exact-image class sets in this export.
These automated results establish structural validity, not perfect semantic
annotation quality. Annotation boxes are never clipped or silently repaired.

Full frequencies for all 85 classes are in `docs/dataset/class_distribution.csv`.
Raw frequency alone is misleading because **3,000 exact redundant images** exist:
3,750 distinct decoded images remain before broader grouping. The final audit
identifies 2,205 related components and 1,062 components crossing source splits.
There are 3,425 perceptual-match edges and 4,586 geometric scene-match edges;
edges overlap with exact/source-ID relationships and are not unique-image counts.

No filenames explicitly marked as augmented were found. Repeated source IDs and
decoded pixels demonstrate duplicate reuse, regardless of the no-augmentation
declaration. Repeated-looking sign tracks, background similarities, source crops
and illustrations remain provenance limitations. Never label them newly collected
Indian road scenes or independently verified physical sign instances.

## Historical v1 duplicate and relationship policy

1. Build exact byte hashes and decoded-pixel hashes; parse original filename
   stems before Roboflow `.rf.` export IDs and explicit augmentation suffixes.
2. Compare **every image pair** for 64-bit DCT pHash distance ≤6 **and** dHash
   distance ≤10. Connect matching images, including transitive relationships.
3. On distinct decoded images, compute SIFT local features at 256×256 for audit
   only. FLANN nearest-descriptor voting proposes candidate pairs; a 0.7 ratio
   test, RANSAC homography, at least 10 inliers, inlier fraction ≥0.65 and spatial
   coverage ≥0.06 retain conservative geometric relationships. Cross-class links
   additionally require matches outside both ground-truth sign boxes, reducing
   false links from shared symbols/watermarks. Seed OpenCV with 42.
4. Union exact, filename, perceptual and geometric edges **before selection**.
   Choose one image per complete connected component for the core.
5. Exclude structurally invalid labels, explicit augmented filenames and exact
   images whose class annotations conflict. No in-scope annotations are dropped.
   The core uses single-class images; mixed-class images are not silently turned
   into backgrounds or relabeled as one class.

The perceptual search is exhaustive under its thresholds. Scene candidate
retrieval is approximate and the source lacks original track IDs: **zero detected
leakage is not a guarantee of independence of all physical signs, sites or video
tracks**. Similar generic signs may be overgrouped; dissimilar views may escape
matching. Six labeled contact sheets covering all 300 selected images were
reviewed at thumbnail resolution for gross class/box issues and domain content.
This is not a certified full-resolution annotation or acquisition-track audit.
No model scores or test predictions informed this process.

## Historical v1 size and class decision

Recommend **a 300-image core** for the initial controlled experiment, not the
nominal 6,750-image export. More data should be added only through a separately
documented, non-overlapping training expansion shared by Models A/C. Larger
counts alone would overstate independent evidence and increase weekend compute.

Choose six classes by independent development/test support. Rank each class by
the smaller of `development_groups / 42.5` and `test_groups / 7.5`, require at
least 30 development and 8 test candidates, and cap speed-limit classes at two
to avoid a speed-only study. IDs break ties. This policy and the data, not model
accuracy, determine the final class set. Highly scene-rich CROSS_ROAD/GAP_IN_MEDIAN
were considered but have only 19/13 eligible development groups respectively
after conservative grouping/source-test quarantine; they do not support the
chosen approximately balanced six-class protocol without repeating groups.

| New ID | Source ID | Source class name | Train | Val | Test | Total |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | COMPULSARY_KEEP_RIGHT | 36 | 8 | 8 | 52 |
| 1 | 42 | NO_ENTRY | 36 | 8 | 8 | 52 |
| 2 | 44 | NO_STOPPING_OR_STANDING | 36 | 8 | 8 | 52 |
| 3 | 47 | PEDESTRIAN_CROSSING | 30 | 7 | 7 | 44 |
| 4 | 64 | SPEED_LIMIT_30 | 36 | 7 | 7 | 50 |
| 5 | 70 | SPEED_LIMIT_80 | 36 | 7 | 7 | 50 |
| | | **Total** | **210** | **45** | **45** | **300** |

Each selected image has one box. Preserve source spellings in the mapping.
The modest 52:44 imbalance reflects real eligible capacity; no oversampling,
fabricated examples or degraded copies were used to force equal counts.

## Splitting, selection and test freeze

The supplied splits leak, so they are not accepted unchanged. Any related group
that touches source test is reserved for final test-only use or excluded; only
an actual source-test member can represent it. All remaining source train/val
groups form a development pool. This preserves the supplied test boundary while
allowing a new group-disjoint train/val split.

Within each pool, select the lexicographically first valid source image per
group. Allocate class counts as evenly as actual capacities allow. Sample across
relative box-area quartiles and image-sharpness terciles, with seed **42**, to
retain framing/clarity variation rather than filtering out difficult examples.
Shuffle the selected development candidates deterministically and assign 45 to
validation, with 210 to training; take 45 source-test representatives separately.
The selection is capacity-balanced, not a claim of population-representative
class/scene frequencies. Source detection mAP cannot be inferred from these counts.

Copy image bytes without resizing/recompression and change only label class-ID
tokens. Coordinate tokens remain unchanged. No synthetic degradation exists.
The final source/test files are never edited by preparation or validation.
`dataset.lock.json` hashes every prepared image, label and YAML. `selection.json`
hashes the manifest, source relationship inventory and lock; the config is
recorded and hashed. A rerun verifies existing files, refuses changes, and does
not overwrite them. A new protocol requires a new output path and explicit
new freeze. Old draft data is excluded from all final manifests and training.

Final test has no model predictions, training use or parameter tuning. Development
may use train/validation only. Later degradation parameters must be frozen using
development examples; paired test variants must leave the clean bytes and boxes
unchanged. Semantic class/track ambiguities cannot be fixed after inspecting test
performance; any necessary data correction requires a documented benchmark revision.

## Reproduce and validate

Download the **version-1 YOLOv8 detection ZIP** from the source link. Place it at
the configured raw path, then run:

```bash
bash run_all.sh --dataset
bash run_all.sh --dataset-validate
```

The first command performs CRC/path/size/symlink-safe staged extraction, source
audit, selection, preparation and tests. It makes no network requests for data
or models. Original ZIP contents remain unchanged. Existing extraction contents
are compared to every archive member and never silently overwritten. A re-export
can change ZIP metadata/checksum; do not claim byte-identical reproduction until
the documented member/image/label fingerprints and manifest agree. Record a new
source revision if actual content differs. A missing source emits the exact
download instructions rather than generating substitutes.

For individual steps, use the existing `.venv`:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py extract
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py audit
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py prepare
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py validate
PYTHONPATH=src .venv/bin/python scripts/dataset_contact_sheets.py
```

Use an **absolute path** to `data/processed/p57_core300_v2/dataset.yaml` when later
invoking Ultralytics. No `path: .` value is emitted, because that can resolve
against Ultralytics' global datasets directory. The real Ultralytics dataset
parser is tested from outside this repository, with automatic download disabled.

Small source/selection/manifest/validation snapshots are retained under
`docs/dataset/` for Git; all source/archive/image/label/generated binary files
remain ignored. `scripts/check_repository_files.py` checks the eligible Git
inventory for secrets/binaries and files above 100 MB. Final checkpoints and
additional datasets will need external versioned download links and hashes;
no trained checkpoints exist in Phase 1.

## GTSRB reference compatibility and limits

The official [benchmark distribution](https://github.com/houbensebastian/GermanTrafficSignBenchmarks)
defines GTSRB as recognition/classification, with original track grouping. Its
ROI boxes on sign crops do not make classification accuracy equivalent to
full-scene YOLO detection mAP. The proposed numeric mapping is recorded in
`docs/dataset/gtsrb_compatibility.csv` for all 85 source classes, and must be
verified against the downloaded official labels before Phase 5 training.

For the core: keep-right, no-entry and the two exact speed limits have candidate
semantic counterparts (GTSRB IDs 38, 17, 1 and 5). No-stopping/standing has no
direct GTSRB counterpart. Pedestrian-crossing is a conditional correspondence
to pedestrian warning (27), not a forced equivalence across all sign designs.
The inspected core contains pedestrian-warning signs, but this does not establish
every source-class image has the same semantics. Candidate mappings are never
used to alter the six detection labels in Phase 1.

Some low-resolution crops resemble GTSRB content. **Actual cross-dataset overlap
has not been measured**, so the future reference cannot be described as an
independent Indian-domain transfer experiment yet. Phase 5 must cross-match the
official GTSRB images/tracks against this frozen core before defining reference
training data and metrics. Unsupported classes remain excluded, not remapped to
incorrect labels. This is a future reference-methodology gate, not fabricated
evidence of independence or a trained reference.

## Proposed Phase 2 start on M3 / 8 GB

`configs/phase2_recommendation.json` records an **unexecuted starting point**:
YOLOv8n, MPS, 416px, batch 4 (reduce to 2/1 if needed), 50 fixed epochs, workers 0,
no RAM image cache, AMP disabled, seed 42 and deterministic behavior. Do a small
train/validation-only feasibility trial after authorization; no speed/memory
budget is claimed as measured. Keep pretrained initialization, architecture,
optimizer, update budget, seed and best-validation-checkpoint rule identical for
A/C. Disable baseline photometric/occlusion augmentation so the E0/E1 difference
is explicit. Add targeted degradation only in E1, with matching training updates.
The final test cannot inform any of these settings.
