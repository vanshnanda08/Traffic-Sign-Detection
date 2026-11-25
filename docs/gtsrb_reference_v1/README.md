# GTSRB four-class crop-recognition reference

The retained 25 November 2026 run scored **91.375% (2,193/2,400)** on the
official final-test crops for GTSRB classes 1, 5, 17 and 38 (speed limit 30,
speed limit 80, no entry and keep right). This is a single-seed classification
result on four classes. W6 measures detection mAP on SDI. The four-class set
omits roundabout and no stopping. The official test labels provide no track
IDs, so independence from the Indian dataset has not been established.

## Recorded protocol

- The [historical script](../../scripts/gtsrb_reference_v1.py) resized the RGB
  crops to 48 × 48 pixels and trained a small CNN from scratch for 15 epochs
  on Apple MPS with seed 42. It used AdamW with a learning rate of 0.002,
  weight decay of 0.0001 and 64-image training batches.
- The historical split contained 5,808 training crops and 1,452 validation
  crops. The run selected epoch 14 by validation accuracy and recorded the
  checkpoint hash before the final-test evaluation. The retained script
  has one final-test evaluation path and stdout records one result.
- The implementation extracted the second component of the training filename,
  which is the frame index, instead of the track ID. The logged 96 and 24
  groups were frame-index groups. An audit using only the archive filenames
  found that all 242 actual sign tracks appeared in both partitions. Both
  tracks listed for exclusion in the freeze receipt also remained in both
  partitions. The validation score therefore does not establish performance
  on unseen tracks. The final-test score is reported as recorded, with this
  limitation.

The experiment results establish the recorded measurement. Written faculty
acceptance of this reference as a substitute was unavailable for verification.

## Evidence and source

- [Saved test metrics and confusion matrix](test_metrics.json). Rows are true
  classes and columns are predicted classes in the listed class order.
- [Original freeze receipt](freeze_receipt.json) and
  [original stdout](run.stdout.txt). The stdout accuracy line uses four
  decimal places. The exact count comes from the matrix diagonal.
- [Release evidence audit](evidence_audit.json). The audit checked the saved
  evidence and archive metadata without training or model inference.

The three source archive hashes are recorded in the freeze receipt. The
repository publishes source code, aggregate results and receipts. The dataset
archives and the trained checkpoint are held separately.

Dataset credit: J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel,
*The German Traffic Sign Recognition Benchmark: A Multi-class Classification
Competition*, IJCNN 2011, pp. 1453–1460.
[Maintainer distribution and citation](https://github.com/houbensebastian/GermanTrafficSignBenchmarks#gtsrb--german-traffic-sign-recognition-benchmark)
