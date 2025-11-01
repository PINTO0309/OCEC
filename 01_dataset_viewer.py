#!/usr/bin/env python3
"""
Hugging Face Dataset Viewer (OpenCV version)
- データセット: MichalMlodawski/closed-open-eyes
- Parquet形式で保存 (data/{split}/train.parquet)
- 既存ファイルがあれば再利用
- OpenCVでランダムサンプルを可視化
"""

import argparse
import io
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
import requests


def resolve_image(image_data):
    """Image-likeオブジェクトをRGBのPIL.Imageに変換して返す"""
    img = None

    if isinstance(image_data, dict):
        file_info = image_data.get("file")

        if isinstance(file_info, Image.Image):
            img = file_info

        elif isinstance(file_info, str) and os.path.exists(file_info):
            try:
                img = Image.open(file_info)
            except Exception as e:
                print(f"[WARN] Could not open local image '{file_info}': {e}")

        elif isinstance(file_info, dict) and "src" in file_info:
            url = file_info["src"]
            try:
                res = requests.get(url, timeout=10)
                res.raise_for_status()
                img = Image.open(io.BytesIO(res.content))
            except Exception as e:
                print(f"[WARN] Could not load from URL '{url}': {e}")

    elif isinstance(image_data, Image.Image):
        img = image_data

    if img is None:
        return None

    if img.mode != "RGB":
        return img.convert("RGB")

    return img.copy()


def visualize_with_opencv(dataset, sample_count: int = 6, window_name: str = "Closed-Open Eyes Samples"):
    """OpenCVを使ってランダムサンプルを可視化"""
    indices = random.sample(range(len(dataset)), min(sample_count, len(dataset)))
    print(f"👁 Showing {len(indices)} random samples with OpenCV...")

    for i, idx in enumerate(indices):
        record = dataset[idx]
        label = record.get("Label", "unknown")
        image_data = record.get("Image_data")
        img = resolve_image(image_data)

        if img is None:
            print(f"[WARN] Skipping index {idx}, no image data found.")
            continue

        # PIL → OpenCV形式（numpy BGR）
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # ラベルテキスト描画
        cv2.putText(img_np, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # 目の反応（座標）からバウンディングボックスを描画
        height, width = img_np.shape[:2]

        def draw_react_box(box, color, title):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return
            x, y, w, h = box
            if w is None or h is None:
                return
            if w <= 0 or h <= 0:
                return
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                return
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            text_pos = (x1, max(0, y1 - 10))
            cv2.putText(img_np, title, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        draw_react_box(record.get("Left_eye_react"), (0, 255, 255), "Left eye")
        draw_react_box(record.get("Right_eye_react"), (255, 0, 0), "Right eye")

        # 表示
        cv2.imshow(window_name, img_np)
        key = cv2.waitKey(0)
        if key == 27:  # ESCで中断
            print("🛑 ESC pressed. Exiting visualization.")
            break

    cv2.destroyAllWindows()


def extract_dataset(dataset, base_outdir: str, split: str):
    """Parquetに含まれる画像とアノテーションをディスクへ展開"""
    extract_root = Path(base_outdir) / "extracted" / split
    total = len(dataset)
    print(f"📤 Extracting {total} samples to {extract_root} ...")

    extracted_count = 0
    created_chunks = set()

    for idx, record in enumerate(dataset):
        img = resolve_image(record.get("Image_data"))
        if img is None:
            print(f"[WARN] Skipping extraction for index {idx}, no image data found.")
            continue

        extracted_count += 1
        base_name = f"{extracted_count:08d}"
        chunk_index = (extracted_count - 1) // 1000 + 1
        chunk_name = f"{chunk_index:08d}"
        chunk_dir = extract_root / chunk_name

        if chunk_name not in created_chunks:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            created_chunks.add(chunk_name)

        image_data = record.get("Image_data")
        ext = ".png"
        if isinstance(image_data, dict):
            filename = image_data.get("filename")
            if filename:
                _, orig_ext = os.path.splitext(filename)
                if orig_ext:
                    ext = orig_ext.lower()

        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in (".jpg", ".png"):
            ext = ".png"

        candidate = chunk_dir / f"{base_name}{ext}"

        save_format = "PNG"
        if ext == ".jpg":
            save_format = "JPEG"

        try:
            img.save(candidate, format=save_format)
        except Exception as e:
            print(f"[WARN] Failed to save image for index {idx}: {e}")
            continue

        annotation = {
            "image_filename": candidate.name,
            "image_id": record.get("Image_id"),
            "label": record.get("Label"),
            "left_eye_react": record.get("Left_eye_react"),
            "right_eye_react": record.get("Right_eye_react"),
            "split": split,
        }
        ann_path = chunk_dir / f"{base_name}.json"
        try:
            with ann_path.open("w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write annotation for index {idx}: {e}")

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"  - Processed {idx + 1}/{total}, saved {extracted_count}")


def main():
    parser = argparse.ArgumentParser(description="Download and visualize MichalMlodawski/closed-open-eyes dataset with OpenCV.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--visualize", action="store_true", help="Visualize random samples with OpenCV")
    parser.add_argument("--sample-count", type=int, default=6, help="Number of samples to visualize")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory (default: ./data)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if parquet exists")
    parser.add_argument("--extract", action="store_true", help="Extract images and annotations to --outdir/extracted/{split}")
    args = parser.parse_args()

    split = args.split
    outdir = os.path.join(args.outdir, split)
    os.makedirs(outdir, exist_ok=True)
    parquet_path = os.path.join(outdir, f"{split}.parquet")

    # --- 既存Parquetチェック ---
    if os.path.exists(parquet_path) and not args.force:
        print(f"✅ {parquet_path} already exists.")
        print("📖 Loading dataset from local Parquet ...")
        ds = Dataset.from_parquet(parquet_path)
    else:
        if args.force:
            print("⚠️  Force mode enabled. Re-downloading dataset...")
        print(f"📦 Downloading dataset split='{split}' ...")
        ds = load_dataset("MichalMlodawski/closed-open-eyes", split=split)
        print(f"✅ Loaded {len(ds)} samples")

        print(f"💾 Saving dataset to {parquet_path} ...")
        ds.to_parquet(parquet_path)
        print(f"✅ Saved parquet: {parquet_path}")

        meta_path = os.path.join(outdir, "info.txt")
        with open(meta_path, "w") as f:
            f.write(f"Dataset: MichalMlodawski/closed-open-eyes\n")
            f.write(f"Split: {split}\n")
            f.write(f"Samples: {len(ds)}\n")
        print(f"🧾 Metadata saved to {meta_path}")

    if args.extract:
        extract_dataset(ds, args.outdir, split)

    if args.visualize:
        visualize_with_opencv(ds, args.sample_count)
    elif not args.extract:
        print("👁 Visualization disabled. Use --visualize to enable.")


if __name__ == "__main__":
    main()
