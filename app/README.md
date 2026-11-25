# P57 local application

Implementation and real CPU/MPS inference are verified on the Apple M3/8 GB Mac.
**W7 final acceptance is blocked:** the ordinary browser upload/comparison/ZIP
flow passed, but the fresh-browser externally blocked flow and remaining browser
error paths have not been fully verified. See [W7 verification](../docs/W7_VERIFICATION.md).
Do not treat this as second-PC verification or completed project submission.

## W8D backend quality audit

The frozen validation-only application diagnostic at confidence.25/IoU.5 produced
A43TP/3FP/3FN and C42TP/10FP/4FN on45images/46labelled boxes. These are micro
fixed-threshold development diagnostics, not mAP or independent held-out evidence.
No threshold/model settings changed to improve them. See
[W8D verification](../docs/W8D_VERIFICATION.md), including a possible duplicate
validation annotation and the models' genuine false positives/misses.

Client revisionw8d.1 fixes optional-counter errors and reaps owned workers on
communication interruption. Worker geometry/prediction remainsw7.1; no weights
changed. Restart a running app normally before verifying revised parent code;
no claim that an existing server refreshed its imported client.35focused cases
passed, but ordinary UI and deferred offline acceptance remain incomplete.

## Install the separate UI environment once

Keep the existing scientific `.venv`, `requirements.txt` and `run_all.sh` intact.
From the project root, using the already-installed Python 3.12.16 and `uv`:

```bash
uv venv --python .venv/bin/python .cache/app-venv
uv pip sync --python .cache/app-venv/bin/python app/requirements-ui.txt
uv pip check --python .cache/app-venv/bin/python
```

Use the corresponding `Scripts/python.exe` paths on Windows. Interpreter lookup
in the launcher/client handles Windows and POSIX layouts, but Windows/Linux and
a second computer have not been tested. The UI lock pins all installed packages,
including Streamlit 1.50.0, Pillow 11.3.0 and psutil 7.2.2. It is deliberately
separate from the scientific runtime. The launcher checks the complete UI lock.
No installation or model download occurs during app operation.

## One normal launcher

```bash
python3 launch_app.py
```

Open `http://127.0.0.1:8501`; stop the owned server with Ctrl+C. `--port 8503`
chooses another unprivileged loopback port. CPU is the default. An explicit MPS
choice is available only when the frozen runtime reports support; there is no
silent device fallback. The server is bound to127.0.0.1, telemetry is disabled,
CORS/XSRF remain enabled and Streamlit's deployment toolbar is hidden. There are
no external fonts/CDNs or remote inference endpoints in the app source.

The launcher supports the macOS-only verification option:

```bash
python3 launch_app.py --offline-sandbox
```

This applies an inherited OS deny-external/allow-loopback network policy to the
server and inference children. It does **not** restrict an already running
browser. A sandboxed external-IP connection was rejected with EPERM while
loopback HTTP and real fixture inference worked. Full fresh-browser offline
acceptance remains open; see the precise scope and failed browser attempts in
the W7 report. Do not call a health check an offline user-flow verification.

## Local checkpoint configuration

`models.json` contains the canonical six classes and the exact replacement A/C
best-checkpoint hashes. Only `path` may be changed to point to a local copy of
the same bytes (absolute paths or paths relative to the project root). A custom
manifest may be selected before startup with `P57_MODEL_MANIFEST`. Do not alter
canonical hashes to admit unrelated weights. Missing/corrupt/wrong files are
rejected before inference, without download fallback. Names inside the loaded
checkpoint must match the canonical classes before prediction. Stable public
checkpoint download links remain an uncompleted distribution gate; none are
invented here. Existing assets stay outside Git.

`P57_INFERENCE_PYTHON` optionally selects an installed scientific interpreter;
otherwise `.venv/bin/python` or `.venv/Scripts/python.exe` is used. It must satisfy
the original scientific dependency pins. Requests use argument arrays and JSON
stdin/stdout, never shell-interpolated upload paths. The inference entry point
does not read research datasets, training logs or multi-GB audit stores. Actual
inference in a separate minimal temporary layout proved that independence; this
does not imply a fresh machine or complete minimal-layout browser verification.

## Workflow and contracts

- **Detect & Compare:** upload JPG/JPEG/PNG; choose A, C or both (default); select
  Original/Fading/Occlusion and mild/medium/strong; mark regions for simulation;
  run, adjust display confidence and download images/JSON/CSV/results ZIP.
- **Measured Research:** authenticates the portable W6 table/metrics/plots using
  `research_manifest.json`. No benchmark is executed or substitute score shown.
  C gains at O2/O3 and loses on clean/fading/O1. The fixed 45-source crop/illustration
  benchmark does not establish general real-road robustness, safety or significance.
- **Help / Limitations:** normalization, predictor/evaluator differences, supported
  classes and research limits. W2/G05 remains blocked.

Image content is decoded and validated, regardless of extension. Limits are
10 MiB and12 megapixels. EXIF orientation is applied **before** region coordinates.
Grayscale/CMYK use Pillow RGB conversion; alpha/palette transparency is composited
on white. Metadata is stripped; calibrated ICC color conversion is not claimed.
Corrupt, animated, empty and oversized images are rejected. New uploads reset
regions/results. Changes to regions, condition, model, device or manifest invalidate
results. The display-confidence filter starts at .25 and regenerates only the
presentation/export from stored raw predictions; downloads do not run inference.

Manual regions are integer pixel rectangles on the normalized upload, not ground
truth or automatic sign segmentation. Add/remove/reset supports multiple regions.
Fading/occlusion requires at least one region. A one-millionth-pixel numerical
inset ensures the frozen floor/ceil rasterizer reconstructs the exact selected
integer rectangles; this is app region conversion, not a benchmark label change.
The unchanged frozen degradation function/configuration processes overlaps once.
The normalized pixel fingerprint supplies the stable upload ID and seed 42 anchors.
Only **one** transformed image is created; its shared pixel hash is included in
every model output. Original normalized pixels are retained separately.

Upload inference uses fixed 448×448 centered, aspect-preserving LetterBox,
scaleup true, stride 32, fill 114, RGB /255 FP32; boxes map to normalized-upload pixel
coordinates. Models run sequentially. Complete predictor NMS disables its wall-clock
break; class-aware IoU.7, raw confidence floor.001, max_det 300, single-label
behavior, no augmentation/filter. W6 instead used validator multi-label behavior
and nominal 416/defaultpad.5 preprocessing with actual448 tensors. Upload results
are not a reproduction of evaluator metrics. No mAP/accuracy is computed without
ground truth. Empty detections/empty display filters are valid.

Padding detections that clip to zero image area remain explicitly flagged in raw JSON/CSV; only positive-area image boxes enter the display filter. This does not discard raw post-NMS evidence or change model/NMS settings.

Annotated PNGs use numbered boxes and separate readable legends. If more than50
displayed detections exist, panels draw the first 50 and declare the cap; JSON/CSV
retain all raw/displayed predictions. The ZIP contains fixed safe filenames only:
normalized/shared PNG, selected model PNGs, JSON, CSV and checksums. It includes
simulation regions/parameters, model hashes, device, actual tensor shapes,
normalization/preprocessing/NMS conventions, input/output hashes and actual child
exit/runtime/resource observations. It contains no checkpoints, datasets or
unrelated files. Checksum JSON covers every other member; it does not attempt a
self-referential checksum. The ZIP's outer checksum is recorded during verification.

Uploads/exports stay in session memory; each worker owns an isolated temporary
directory removed on return/failure. No upload is added to the research dataset.
Verification deliberately saved controlled development-source fixtures and exports
under ignored `results/w7_verification_v1`; that is not default user-data retention.
The only executable UI operations are capabilities, transform and inference;
no training, benchmark, package-install or publication action is exposed.

## Verification and attribution

Run focused tests with `PYTHONPATH=app:src .venv/bin/python -m pytest tests/test_app.py`.
`scripts/verify_app_runtime.py` is an explicit development-fixture check, with
versioned output paths; it does not evaluate the held-out benchmark. Verification
helpers `deny_external_web.py` and `prepare_offline_browser.py` are test-only,
not production app launchers; browser proxy verification is still incomplete.

UI configuration follows the [official Streamlit reference](https://docs.streamlit.io/develop/api-reference/configuration/config.toml).
The compatible older release is listed in [official 2025 release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025).
The browser proxy design follows [Chromium's primary proxy documentation](https://chromium.googlesource.com/chromium/src/+/HEAD/net/docs/proxy.md).
P57 reuses Ultralytics/PyTorch/OpenCV/Pillow/Streamlit and the public SDI v1 dataset;
see project credits/provenance. Public data substitution is student-reported
faculty approval. AI assisted app implementation, tests and documentation; required
team contribution review/sign-off remains pending.
