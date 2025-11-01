# OCEC
open closed eyes classification

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
