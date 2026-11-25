# Obtain exact inputs through authorized channels

Acquire the SDI Indian Traffic Sign **version 1 YOLOv8 detection export**
from the publisher or another authorized custodian. Authenticate the
frozen export SHA-256
`d4ce92dfd534681f4f0a70de3a217a4a79e7146382450c3c030888244306a762`.
A later or repackaged export is rejected. Keep the publisher's attribution
and applicable third-party rights. [Exact version](https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1)

Obtain the two exact fine-tuned P57 checkpoints from an authorized rights
holder or approved private transfer.

- Model A SHA-256 `15ed8470368fc535eed861d89dbee09b55977f1f935b40d41b24e052a6d6e4da`
- Model C SHA-256 `cb918cad486d0fc6a2eb4c4886cd446bb4aef430d2f6e3581ccc2abff84c8091`

The public Ultralytics initialization cannot stand in for either of these
files. The Git tree does not include the source ZIP, checkpoints or images.

After the pinned environment is installed, run from a fresh checkout.
Use the separately authorized input paths from this Mac.

```bash
.venv/bin/python scripts/install_authorized_inputs.py \
  --dataset "/path/to/authorized/SDI-v1-YOLOv8.zip" \
  --model-a "/path/to/authorized/A.pt" \
  --model-c "/path/to/authorized/C.pt"
.venv/bin/python scripts/release_assets.py verify
bash run_all.sh
```

The installer reads the exact 45 validation and 45 test image/label pairs
from the authenticated export, applies the frozen class remap and verifies
the resulting bytes against the release manifest. It copies the two
checkpoints without loading them. It uses the same default paths as the
benchmark and refuses occupied targets. It performs no downloads, inference
or training. The benchmark runs only when `bash run_all.sh` is invoked.

If acquisition rights or exact bytes are unavailable, record the blocker.
`scripts/release_assets.py install` also accepts the unchanged canonical
combined archive through a separately authorized private transfer.
An authorized local or private transfer is independent of public retrieval.
