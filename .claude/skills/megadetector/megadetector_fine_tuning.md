# Fine-tuning MegaDetectorV6 RT-DETR with plain ultralytics

**The AGPL-licensed `MDV6-rtdetr-c.pt` can be loaded and fine-tuned with `from ultralytics import RTDETR`, but the Apache-licensed `.pth` weights cannot.** The critical distinction is that PytorchWildlife ships two entirely separate RT-DETR implementations under different licenses — one built on ultralytics (AGPL, `.pt` format), the other on a non-ultralytics codebase (Apache 2.0, `.pth` format). This split exists specifically to let commercial users avoid AGPL contamination. The officially recommended fine-tuning path is PytorchWildlife's own `PW_FT_detection` module, which wraps ultralytics internally. Fine-tuning RT-DETR through ultralytics is officially supported but carries several well-documented pitfalls that can derail training.

## Three RT-DETR weight files, two incompatible codebases

PytorchWildlife hosts all MDv6 weights on a single Zenodo record (DOI: 10.5281/zenodo.15398270, version v22, published May 13, 2025). The RT-DETR–specific files are:

| File | Size | Format | License | Loaded via |
|------|------|--------|---------|------------|
| `MDV6-rtdetr-c.pt` | **66.1 MB** | Ultralytics `.pt` | AGPL-3.0 | `MegaDetectorV6(version='MDV6-rtdetr-c')` |
| `MDV6-apa-rtdetr-c.pth` | **322.4 MB** | PyTorch `.pth` | Apache 2.0 | `MegaDetectorV6Apache(version='MDV6-apa-rtdetr-c')` |
| `MDV6-apa-rtdetr-e.pth` | **1.2 GB** | PyTorch `.pth` | Apache 2.0 | `MegaDetectorV6Apache(version='MDV6-apa-rtdetr-e')` |

Direct download URLs follow the pattern `https://zenodo.org/records/15398270/files/<filename>?download=1`. There is **no** AGPL `MDV6-rtdetr-e.pt` — the extra-large RT-DETR exists only under the Apache license. All models detect three classes: animal, person, and vehicle at **1280×1280** input resolution.

The **~5× size gap** between `MDV6-rtdetr-c.pt` (66 MB) and `MDV6-apa-rtdetr-c.pth` (322 MB) is the clearest evidence these are different codebases. The ultralytics `.pt` format bundles architecture YAML, training arguments, and a compressed state dict into one file, while the `.pth` is a standard PyTorch state dict — likely from the original lyuwenyu/RT-DETR implementation (the CVPR 2024 paper's official Apache-licensed codebase). The different layer naming conventions and serialization approaches make these two weight formats mutually incompatible.

## The AGPL weights were trained with ultralytics; the Apache weights were not

Multiple pieces of evidence confirm the training lineage of each variant:

**Evidence that `MDV6-rtdetr-c.pt` was trained using ultralytics:**
- The file uses ultralytics' proprietary `.pt` serialization format, which embeds the model YAML config and training hyperparameters alongside weights.
- In the CameraTraps source code, the `MegaDetectorV6` class inherits from `YOLOV8Base`, located in `PytorchWildlife/models/detection/ultralytics_based/megadetectorv6.py`, confirming it plugs directly into the ultralytics inference pipeline.
- The v1.2.0 release notes state explicitly: **"Currently the fine-tuning is based on Ultralytics with AGPL. We will release MIT versions in the future."**
- The earliest MDv6 release (v1.1.0, November 2024) used the naming convention "MDv6-ultralytics-yolov9-compact."

**Evidence that `MDV6-apa-rtdetr-*.pth` uses a different implementation:**
- Apache 2.0 licensing specifically requires avoiding ultralytics' AGPL-licensed code — the entire point of the separate Apache variant is AGPL avoidance.
- A separate `MegaDetectorV6Apache` class (imported from the standalone `pw_detection` module, not from `PytorchWildlife.models`) handles loading. The source code for this class is not fully documented in the public API reference, suggesting it was added more recently and may wrap the lyuwenyu/RT-DETR PyTorch implementation.
- Standard PyTorch `.pth` format rather than ultralytics `.pt`.

## Loading `MDV6-rtdetr-c.pt` with plain ultralytics works — with version caveats

Because `MDV6-rtdetr-c.pt` is a native ultralytics checkpoint, this code is expected to succeed:

```python
from ultralytics import RTDETR
model = RTDETR("MDV6-rtdetr-c.pt")
model.info()  # Should show 3-class RT-DETR architecture
```

PytorchWildlife itself does exactly this under the hood — `YOLOV8Base` wraps `ultralytics.YOLO()`, and ultralytics auto-detects RT-DETR architecture from the embedded config in the `.pt` file. You could also use `YOLO("MDV6-rtdetr-c.pt")` and ultralytics will route to the correct internal model class, though using the explicit `RTDETR()` constructor is cleaner.

**No public GitHub issues or forum posts were found** where someone specifically reported loading `MDV6-rtdetr-c.pt` with plain ultralytics. This is a gap in community documentation — the PytorchWildlife ecosystem encourages users to work through its own API rather than dropping to raw ultralytics.

However, a **critical regression bug** (ultralytics issue #20627) broke RT-DETR fine-tuning in **ultralytics v8.3.133** (May 2025). The `load()` method in `nn/tasks.py` assumed all models have a YOLO-style first convolution key (`model.0.conv.weight`), which RT-DETR's ResNet backbone does not. This caused a `KeyError` when `model.train()` attempted to load weights. The issue is labeled "fixed" in subsequent versions. **Anyone attempting this workflow must use an ultralytics version after this fix.**

The Apache `.pth` weights **cannot** be loaded with `RTDETR()` from ultralytics — the layer names, architecture, and format are entirely different. Attempting `RTDETR("MDV6-apa-rtdetr-c.pth")` will fail.

## Fine-tuning RT-DETR in ultralytics is supported but fragile

Ultralytics officially supports Train, Val, Predict, and Export modes for RT-DETR. The standard fine-tuning workflow would be:

```python
from ultralytics import RTDETR
model = RTDETR("MDV6-rtdetr-c.pt")
results = model.train(data="my_dataset.yaml", epochs=100, imgsz=1280, amp=False)
```

However, the community has documented **six significant gotchas** that apply to any RT-DETR fine-tuning in ultralytics, all of which would affect MDv6 weights:

- **AMP causes NaN outputs.** The ultralytics source code itself warns that automatic mixed precision "can lead to NaN outputs and may produce errors during bipartite graph matching." Setting `amp=False` is strongly recommended, at the cost of higher VRAM usage.
- **NaN loss after ~50 epochs with AdamW** (issue #7594). RT-DETR's transformer decoder is sensitive to learning rate — start lower than YOLO defaults.
- **mAP decreasing while training loss decreases** (issue #20745). Validation metrics diverge from training metrics, suggesting RT-DETR overfits faster than YOLO models and requires careful early stopping.
- **`model.tune()` does not work** (issue #14388). The hyperparameter tuning method internally invokes `yolo train`, not the RT-DETR trainer.
- **The `model.0.conv.weight` KeyError** (issue #20627, now fixed). Pin your ultralytics version to one after this fix.
- **`F.grid_sample` rejects `deterministic=True`** — set `deterministic=False` when training.

Training RT-DETR from scratch (without pretrained backbone weights) yields near-zero accuracy (issues #19530, #6924), so starting from pretrained weights like MDv6 is effectively mandatory for acceptable results. The ultralytics `RTDETRTrainer` handles head adaptation when fine-tuning to a different number of classes.

## The Apache and MIT variants serve commercial licensing needs

The three-tier licensing structure of MDv6 exists to address a persistent community pain point: ultralytics' AGPL license makes its outputs unusable in proprietary commercial products without an enterprise license.

The **MIT-licensed models** (`MDV6-mit-yolov9-c.ckpt`, `MDV6-mit-yolov9-e.ckpt`) use a custom YOLO implementation in `.ckpt` format, loaded via `MegaDetectorV6MIT`. The **Apache-licensed models** (`MDV6-apa-rtdetr-c.pth`, `MDV6-apa-rtdetr-e.pth`) use what is almost certainly the lyuwenyu/RT-DETR PyTorch codebase, loaded via `MegaDetectorV6Apache`. Both are imported from `pw_detection`, a separate module from the main `PytorchWildlife` package, with its own loading infrastructure that does not touch ultralytics code. These classes became available in PytorchWildlife **v1.2.4** (latest release is v1.2.4.2 on PyPI).

Fine-tuning the Apache `.pth` weights would require using the lyuwenyu/RT-DETR training scripts (or equivalent), not ultralytics. **No official fine-tuning pathway for the Apache weights has been documented** by the PytorchWildlife team — the `PW_FT_detection` module currently only wraps ultralytics.

## The official fine-tuning path goes through PW_FT_detection

PytorchWildlife's recommended approach is its own **`PW_FT_detection/`** module, released in v1.2.0 (January 2025). This module:

- Wraps ultralytics' training pipeline under the hood (and therefore carries AGPL licensing).
- Supports fine-tuning from **any** released MDv6 pretrained checkpoint, including `MDV6-rtdetr-c`.
- Includes dataset preparation utilities for converting custom annotations into the required format.
- Produces fine-tuned weights that can be reloaded into the PytorchWildlife inference pipeline via the `custom_weight_loading.ipynb` demo notebook and the Gradio web app.

The workflow documented in the repo is: (1) prepare data using PW_FT_detection's built-in utilities, (2) fine-tune from a pretrained MDv6 checkpoint, (3) load the resulting custom weights back into PytorchWildlife via `MegaDetectorV6(weights="path/to/custom.pt")`. The release notes from v1.1.0 confirm: "You can now load custom weights you fine-tuned on your own datasets using the finetuning module directly in the Pytorch-Wildlife pipeline."

The team has stated MIT-licensed fine-tuning modules are planned but **not yet released**. No separate fine-tuning tutorial exists specifically for RT-DETR — the detection fine-tuning module handles all MDv6 architectures uniformly through ultralytics.

## Conclusion

For the specific question of whether `from ultralytics import RTDETR` can load and fine-tune `MDV6-rtdetr-c.pt`: **yes, it can**, because these weights are native ultralytics checkpoints. The `.train()` method is officially supported for RT-DETR. The practical barriers are not format compatibility but rather training stability — AMP-induced NaN losses, learning rate sensitivity, and a now-fixed `KeyError` regression in ultralytics v8.3.133. The Apache `.pth` weights are a completely separate model family that cannot be loaded by ultralytics at all. For production use, the safest path remains PytorchWildlife's `PW_FT_detection` module, which handles ultralytics' RT-DETR quirks and provides an integrated pipeline from data preparation through inference deployment.