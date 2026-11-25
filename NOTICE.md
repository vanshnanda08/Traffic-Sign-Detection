# P57 license and attribution

Copyright (C) 2026 P57 Project, for the project-owned material.

Project-owned source code, scripts, tests and build and installation materials
in this revision are licensed under the GNU Affero General Public License,
version 3 only (AGPL-3.0-only). See [LICENSE](LICENSE). The name P57 Project
identifies the project in Git. The license adds no third-party permissions.

## Ultralytics

P57 uses Ultralytics YOLOv8n and depends on Ultralytics 8.3.228. P57's
validator, predictor, training and deployment adapters and scripts are
provided in source form. The pinned upstream implementation and its notices
remain available from the [v8.3.228 source](https://github.com/ultralytics/ultralytics/tree/v8.3.228)
and the [retained upstream license](docs/release_w8f_v1/upstream/ULTRALYTICS_AGPL-3.0.txt).
The original initialization is separately provided by
[Ultralytics Assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0).
The initial model is distinct from the canonical fine-tuned A and C weights.
Model weights and asset bundles are absent from Git. A separate model
distribution must carry its applicable license and corresponding source.

## Dataset input and attribution

Credit SDI, *Indian Traffic Sign*, Roboflow Universe v1, generated 8 June 2023.
The publisher lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
P57 selected a subset, remapped class IDs and reassigned development splits
while preserving the source export image bytes. The frozen evaluation
manifest embeds 90 remapped annotation texts, alongside file hashes, to
authenticate the exact input and regeneration. Those annotation texts retain
their upstream provenance and applicable terms. This project grant extends
only to project-owned material. The version listing gives no independent
assurance about third-party stock imagery or rights.

[SDI version 1](https://universe.roboflow.com/sdi/indian-traffic-sign/dataset/1) ·
[Dataset protocol](docs/DATASET_PROTOCOL.md) ·
[Asset notices](docs/release_v1/ASSET_NOTICES.md)

## Other software and generated results

Python, PyTorch, Streamlit and other installed packages keep their respective
licenses. They are obtained from their upstream distributions under the
[scientific](docs/release_v1/scientific_dependency_credits.json) and
[UI](docs/release_v1/ui_dependency_credits.json) pins and notices. The renderer
uses a system-installed Arial font, which is outside this repository.
The project-generated W6 tables, metrics and plots identify their frozen data
and checkpoint inputs. The app disclosures and limitations are in the README.
