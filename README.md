# OCEC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17505461.svg)](https://doi.org/10.5281/zenodo.17505461) ![GitHub License](https://img.shields.io/github/license/pinto0309/ocec) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/ocec)

Open closed eyes classification. Ultra-fast wink and blink estimation model.

In the real world, attempting to detect eyes larger than 20 pixels high and 40 pixels wide is a waste of computational resources.

https://github.com/user-attachments/assets/2ae9467f-a67f-447e-8704-d16efacdacf1

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|112 KB|0.9924|0.16 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_p.onnx)|
|N|176 KB|0.9933|0.25 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_n.onnx)|
|S|494 KB|0.9943|0.41 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_s.onnx)|
|C|875 KB|0.9947|0.49 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_c.onnx)|
|M|1.7 MB|0.9949|0.57 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_m.onnx)|
|L|6.4 MB|0.9954|0.80 ms|[Download](https://github.com/PINTO0309/OCEC/releases/download/onnx/ocec_l.onnx)|

## Setup

```bash
git clone https://github.com/PINTO0309/OCEC.git && cd OCEC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```
## Inference

```bash
uv run python demo_ocec.py \
-v 0 \
-m deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
-om ocec_l.onnx \
-ep cuda

uv run python demo_ocec.py \
-v 0 \
-m deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
-om ocec_l.onnx \
-ep tensorrt
```

## Dataset Preparation
```bash
uv run python 01_dataset_viewer.py --split train
uv run python 01_dataset_viewer.py --split train --visualize
```
```bash
uv run python 01_dataset_viewer.py --split train --extract
```
```bash
uv run python 02_real_data_size_hist.py \
-v real_data/open.mp4 \
-oep tensorrt \
-dvw

# [Eye Analysis] open
#   Total frames processed: 930
#   Frames with Eye detections: 930
#   Frames without Eye detections: 0
#   Frames with ≥3 Eye detections: 2
#   Total Eye detections: 1818
#   Histogram PNG: output_eye_analysis/open_eye_size_hist.png
#     Width  -> mean=20.94, median=22.00
#     Height -> mean=11.39, median=11.00

uv run python 02_real_data_size_hist.py \
-v real_data/closed.mp4 \
-oep tensorrt \
-dvw

# [Eye Analysis] closed
#   Total frames processed: 1016
#   Frames with Eye detections: 1016
#   Frames without Eye detections: 0
#   Frames with ≥3 Eye detections: 38
#   Total Eye detections: 1872
#   Histogram PNG: output_eye_analysis/closed_eye_size_hist.png
#     Width  -> mean=15.25, median=14.00
#     Height -> mean=8.17, median=7.00
```

```
Considering practical real-world sizes,
I adopt an input resolution of height x width = 24 x 40.
```

```bash
uv run python 03_wholebody34_data_extractor.py \
-ea \
-m deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx \
-oep tensorrt

# Eye-only detection summary
#   Total images: 131174
#   Images with detection: 130596
#   Images without detection: 578
#   Images with >=3 detections: 1278
#   Crops per label:
#     closed: 134522
#     open: 110796

# Eye-only detection summary
#   Total images: 144146
#   Images with detection: 143364
#   Images without detection: 782
#   Images with >=3 detections: 1221
#   Crops per label:
#     closed: 136347
#     open: 135319
```

### Data sample

|Label|ex1|ex2|ex3|ex4|ex5|ex6|ex7|ex8|ex9|ex10|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|open|<img width="23" height="9" alt="open1_00000001_1" src="https://github.com/user-attachments/assets/e9e3d5f8-3725-4a02-bad2-6d782bc53df3" />|<img width="27" height="9" alt="open1_00000001_2" src="https://github.com/user-attachments/assets/1319e718-c7d8-4dc6-ac8a-d0e935e09cff" />|<img width="23" height="11" alt="open2_00002606_1" src="https://github.com/user-attachments/assets/591b76f9-335a-4ce5-bb02-0d2e18f890a5" />|<img width="19" height="11" alt="open2_00002606_2" src="https://github.com/user-attachments/assets/dc50b131-e9e1-4008-9cf6-64fb72c82381" />|<img width="61" height="37" alt="open3_00000289_1" src="https://github.com/user-attachments/assets/7d5267c3-cd00-403d-b651-f876991f330a" />|<img width="51" height="35" alt="open3_00000289_2" src="https://github.com/user-attachments/assets/35b36785-89b7-4d5d-b74e-c1f71ff3ac32" />|<img width="12" height="10" alt="open4_00000869_1" src="https://github.com/user-attachments/assets/83af4630-a97c-4d22-bd62-772399485fdc" />|<img width="18" height="12" alt="open4_00000869_2" src="https://github.com/user-attachments/assets/5ac9c003-6690-4f2e-a8a0-0992414438c9" />|<img width="26" height="14" alt="open5_00001725_1" src="https://github.com/user-attachments/assets/93dc4094-26ba-42b3-bd66-13d6f67e6f79" />|<img width="35" height="14" alt="open5_00001725_2" src="https://github.com/user-attachments/assets/6ec669b8-ae53-4748-bdb5-c884cb5f4e2e" />|
|closed|<img width="22" height="9" alt="closed_00000414_1" src="https://github.com/user-attachments/assets/ae69e8e6-3d9e-4479-893d-7c0aa236334c" />|<img width="23" height="9" alt="closed_00000414_2" src="https://github.com/user-attachments/assets/6ef22ac9-e7c6-464b-a27d-6b41a8e43322" />|<img width="20" height="13" alt="closed_00000962_1" src="https://github.com/user-attachments/assets/74fc2c2e-66ae-4076-ad24-a405b45dde1f" />|<img width="24" height="13" alt="closed_00000962_2" src="https://github.com/user-attachments/assets/8c2a687e-d8b8-4933-ba32-b311f67aa79e" />|<img width="17" height="8" alt="closed_00001317_1" src="https://github.com/user-attachments/assets/8a588ac3-5415-4405-91b8-98a5540f4d21" />|<img width="22" height="9" alt="closed_00001317_2" src="https://github.com/user-attachments/assets/85bd54f1-f790-4501-ac4b-3e07395539fc" />|<img width="23" height="11" alt="closed_00001502_1" src="https://github.com/user-attachments/assets/4f4157ef-2719-4b6f-a35c-da96a2efe018" />|<img width="19" height="7" alt="closed_00001502_2" src="https://github.com/user-attachments/assets/edc2cf1c-c2e2-4809-b9ee-e5675d69cdd4" />|<img width="20" height="8" alt="closed_00001752_1" src="https://github.com/user-attachments/assets/b8836987-eaab-4be1-8704-f428b48a8727" />|<img width="23" height="6" alt="closed_00001752_2" src="https://github.com/user-attachments/assets/a7890864-5870-4c34-9e97-8065f44375c5" />|

<img width="800" alt="open_eye_size_hist" src="https://github.com/user-attachments/assets/806dc7ec-4132-45a5-ae38-498eb848cf86" />
<img width="800" alt="closed_eye_size_hist" src="https://github.com/user-attachments/assets/4806ae1c-0654-4b79-919f-a113235d6441" />

```bash
uv run python 04_dataset_convert_to_parquet.py \
--annotation data/cropped/annotation.csv \
--output data/dataset.parquet \
--train-ratio 0.8 \
--seed 42 \
--embed-images

#Split summary: {'train_total': 196253, 'train_closed': 107617, 'train_open': 88636, 'val_total': 49065, 'val_closed': 26905, 'val_open': 22160}
#Saved dataset to data/dataset.parquet (245318 rows).

# Split summary: {'train_total': 217332, 'train_closed': 109077, 'train_open': 108255, 'val_total': 54334, 'val_closed': 27270, 'val_open': 27064}
# Saved dataset to data/dataset.parquet (271666 rows).
```

Generated parquet schema (`split`, `label`, `class_id`, `image_path`, `source`):
- `split`: `train` or `val`, assigned with an 80/20 stratified split per label.
- `label`: string eye state (`open`, `closed`); inferred from filename or class id.
- `class_id`: integer class id (`0` closed, `1` open) maintained from the annotation.
- `image_path`: path to the cropped PNG stored under `data/cropped/...`.
- `source`: `train_dataset` for `000000001`-prefixed folders, `real_data` for `100000001`+, `unknown` otherwise.
- `image_bytes` *(optional)*: raw PNG bytes for each crop when `--embed-images` is supplied.

Rows are stratified within each label before concatenation, so both splits keep similar open/closed proportions. Class counts per split are printed when the conversion script runs.

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss`, `pos_weight`, and a `WeightedRandomSampler` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `ocec_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `ocec_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/ocec_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
uv run python -m ocec train \
--data_root data/dataset.parquet \
--output_dir runs/ocec \
--epochs 50 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--image_size 24x40 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
uv run python -m ocec train \
--data_root data/dataset.parquet \
--output_dir runs/ocec_is_s \
--epochs 50 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--image_size 24x40 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
uv run python -m ocec train \
--data_root data/dataset.parquet \
--output_dir runs/ocec_convnext \
--epochs 50 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--image_size 24x40 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 3x2 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `ocec_epoch_*.pt`, the latest 10 `ocec_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/ocec/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/ocec
  ```

### ONNX Export

```bash
uv run python -m ocec exportonnx \
--checkpoint runs/ocec_is_s/ocec_best_epoch0049_f1_0.9939.pt \
--output ocec_s.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_open` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Arch

<img width="300" alt="ocec_p" src="https://github.com/user-attachments/assets/fa54cf38-0fd4-487a-bf9e-dfbc5401a389" />

## Ultra-lightweight classification model series
1. [VSDLM: Visual-only speech detection driven by lip movements](https://github.com/PINTO0309/VSDLM) - MIT License
2. [OCEC: Open closed eyes classification. Ultra-fast wink and blink estimation model](https://github.com/PINTO0309/OCEC) - MIT License
3. [PGC: Ultrafast pointing gesture classification](https://github.com/PINTO0309/PGC) - MIT License
4. [SC: Ultrafast sitting classification](https://github.com/PINTO0309/SC) - MIT License
5. [PUC: Phone Usage Classifier is a three-class image classification pipeline for understanding how people
interact with smartphones](https://github.com/PINTO0309/PUC) - MIT License

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025ocec,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/OCEC},
  month     = {10},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17505461},
  url       = {https://github.com/PINTO0309/ocec},
  abstract  = {Open closed eyes classification. Ultra-fast wink/blink estimation model.},
}
```

## Acknowledgements
- https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes: Open Data Commons Attribution License (ODC-By) v1.0
  ```bibtex
  @misc{open_closed_eyes2024,
    author = {Michał Młodawski},
    title = {Open and Closed Eyes Dataset},
    month = July,
    year = 2024,
    url = {https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes},
  }
  ```
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34 - Apache 2.0
