# OCEC
open closed eyes classification

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|P|112 KB|0.9911|0.16 ms|[Download]()|
|N|176 KB|0.9922|0.25 ms|[Download]()|
|S|494 KB|0.9939|0.41 ms|[Download]()|
|M|1.7 MB|0.9943|0.57 ms|[Download]()|
|L|6.4 MB|0.9948|0.80 ms|[Download]()|

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
-vm ocec_l.onnx \
-ep cuda

uv run python demo_ocec.py \
-v 0 \
-m deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
-vm ocec_l.onnx \
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
```

```bash
uv run python 04_dataset_convert_to_parquet.py \
--annotation data/cropped/annotation.csv \
--output data/dataset.parquet \
--train-ratio 0.8 \
--seed 42 \
--embed-images

#Split summary: {'train_total': 196253, 'train_closed': 107617, 'train_open': 88636, 'val_total': 49065, 'val_closed': 26905, 'val_open': 22160}
#Saved dataset to data/dataset.parquet (245318 rows).
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
- Only samples with `class_id` equal to 1 (closed) or 2 (open) are used. Label 0 (unknown) is dropped, and the remaining labels are remapped to 0 (closed) and 1 (open).
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss`, `pos_weight`, and a `WeightedRandomSampler` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `ocec_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `ocec_best_epoch0004_f10.9321.pt` (also latest 10).
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

- Outputs include the latest 10 `ocec_epoch_*.pt`, the latest 10 `ocec_best_epochXXXX_f1YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
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

uv run python -m ocec webcam_onnx \
--model ocec.onnx \
--camera_index 0 \
--provider cuda \
--detector_provider tensorrt
```

- The saved graph exposes `images` as input and `prob_open` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Acknowledgements
- https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes
  ```bibtex
  @misc{open_closed_eyes2024,
    author = {Michał Młodawski},
    title = {Open and Closed Eyes Dataset},
    month = July,
    year = 2024,
    url = {https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes},
  }
  ```
