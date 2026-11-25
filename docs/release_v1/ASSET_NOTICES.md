# P57 external reproduction assets — attribution and distribution conditions

Prepared 22 November 2026. Local transfer candidate only; public hosting, release
license adoption and corresponding-source availability have not been verified.

## Dataset

Credit SDI, Indian Traffic Sign, Roboflow Universe v1 (generated 8 June 2023):
https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1
The version page was checked on 22 November 2026 and lists CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
Authoritative legal code: https://creativecommons.org/licenses/by/4.0/legalcode.en
Retain credit, license link, and modification information; do not imply endorsement
or add restrictions contrary to the license. The license does not warrant all
third-party privacy/publicity rights. This records the publisher's representation,
not independent ownership verification of each upstream photograph/illustration.

P57 selected/remapped a 300-image six-class subset. This archive contains only the
45 validation and 45 held-out originals with the exact frozen P57 labels (class
IDs remapped during dataset preparation). Original exported image bytes are
unchanged: no new encoding, selection, annotations or predictions. See the pinned
release manifest, dataset lock, selection manifest and source provenance in the
repository. Upstream auto-orientation/resizing is not described as camera-original
capture. Faculty approval of public-data substitution is student-reported.

## Models and implementation

Canonical replacement A/C are P57 fine-tuned YOLOv8n checkpoints, using Ultralytics
pretrained YOLOv8n initialization; they are not trained from scratch. Model hashes
and exact training/protocol records are retained in the repository. Upstream:
https://github.com/ultralytics/ultralytics/tree/v8.3.228
https://github.com/ultralytics/assets
https://docs.ultralytics.com/models/yolov8/
Pinned source license:
https://github.com/ultralytics/ultralytics/blob/v8.3.228/LICENSE
Current vendor terms: https://www.ultralytics.com/license

Ultralytics publishes AGPL-3.0 and commercial licensing routes, with academic
open-source use under AGPL obligations. Treat the derived checkpoints/project as
subject to applicable Ultralytics conditions; this notice does not grant a new
permissive model license. Before public upload, the team must confirm its rights,
adopt an appropriate project/model license, retain notices and provide applicable
corresponding source under those terms (or document a valid alternative license).
No enterprise license or team-wide license decision is asserted here. Public
redistribution is conditional, not cleared by local packaging or Git committing.

Dependencies retain their own licenses; see docs/THIRD_PARTY_NOTICES.md and the
installed-distribution license inventory. Download the original exact dataset
version from its publisher if redistribution cannot be cleared; use the locked
source IDs/hashes and P57 labels to verify matches. A changing export is not an
acceptable substitute. Exact trained weights have no equivalent upstream download;
without authorized exact-weight transfer, reproduction remains blocked. Retraining
or using generic weights would be a different experiment and is not a workaround.

## Exclusions

No GTSRB material or claimed GTSRB clearance; W2/G05 is separately blocked.
No fonts, full raw export, training images, runtime environment, pretrained source
checkpoint, training stores, reports or submission ZIP are included. Arial is
proprietary: use a legitimately installed local copy; never redistribute it here.
Local package checks establish integrity only, not public publication, rights
clearance, independent reproduction or application/offline acceptance.
