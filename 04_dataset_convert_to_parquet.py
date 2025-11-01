#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cropped eye dataset into a single parquet file with "
            "balanced train/validation splits."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/cropped/annotation.csv"),
        help="Path to annotation CSV (default: data/cropped/annotation.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cropped/eye_dataset.parquet"),
        help="Destination parquet file (default: data/cropped/eye_dataset.parquet).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (between 0 and 1). Default: 0.8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling prior to splitting. Default: 42.",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed raw image bytes into the parquet output (column: image_bytes).",
    )
    args = parser.parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be in the (0, 1) interval.")
    return args


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    df = pd.read_csv(
        annotation_path,
        header=None,
        names=("image_path", "class_id"),
        dtype={"image_path": str, "class_id": int},
    )
    if df.empty:
        raise ValueError(f"No entries loaded from {annotation_path}.")
    df["image_path"] = df["image_path"].apply(lambda p: str(Path(p)))
    df["class_id"] = df["class_id"].astype(int)
    df["label"] = df.apply(lambda row: infer_label(row["image_path"], row["class_id"]), axis=1)
    df["source"] = df["image_path"].apply(determine_source)
    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    missing = df.loc[~df["exists"], "image_path"].tolist()
    if missing:
        raise FileNotFoundError(
            "Some image files are missing. "
            "Examples:\n" + "\n".join(missing[:10])
        )
    df.drop(columns=["exists"], inplace=True)
    return df


def embed_image_bytes(df: pd.DataFrame, column_name: str = "image_bytes") -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    def read_bytes(path_str: str) -> bytes:
        return Path(path_str).read_bytes()

    iterator: Iterable[str] = df["image_path"]
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Embedding images", unit="img")

    df = df.copy()
    df[column_name] = [read_bytes(path) for path in iterator]
    return df


def infer_label(path_str: str, class_id: int) -> str:
    stem = Path(path_str).stem.lower()
    if "open" in stem:
        return "open"
    if "closed" in stem:
        return "closed"
    return "open" if class_id == 1 else "closed"


def determine_source(path_str: str) -> str:
    path = Path(path_str)
    try:
        folder_name = path.parts[path.parts.index("cropped") + 1]
    except (ValueError, IndexError):
        return "unknown"
    if folder_name.isdigit() and int(folder_name) >= 100000000:
        return "real_data"
    if folder_name.isdigit():
        return "train_dataset"
    return "unknown"


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_rows: List[pd.Series] = []
    val_rows: List[pd.Series] = []

    for label, group in df.groupby("label"):
        records = list(group.to_dict(orient="records"))
        rng.shuffle(records)
        n_total = len(records)
        if n_total == 0:
            continue
        n_train = int(n_total * train_ratio)
        if n_train == 0 and n_total > 1:
            n_train = 1
        if n_total - n_train == 0 and n_total > 1:
            n_train -= 1
        train_subset = records[:n_train] if n_train > 0 else []
        val_subset = records[n_train:]
        train_rows.extend(train_subset)
        val_rows.extend(val_subset)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df


def validate_balances(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    def describe(df: pd.DataFrame, split: str) -> Dict[str, int]:
        counter = Counter(df["label"])
        total = int(counter.total())
        description = {f"{split}_total": total}
        for label, count in counter.items():
            description[f"{split}_{label}"] = count
        return description

    train_counts = describe(train_df, "train")
    val_counts = describe(val_df, "val")
    summary = train_counts | val_counts
    print("Split summary:", summary)


def main() -> None:
    args = parse_args()
    df = load_annotations(args.annotation)
    if args.embed_images:
        df = embed_image_bytes(df)
    train_df, val_df = stratified_split(df, args.train_ratio, args.seed)

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    column_order = ["split", "label", "class_id", "image_path", "source"]
    if args.embed_images:
        column_order.append("image_bytes")
    combined_df = combined_df[column_order].sort_values(["split", "label", "image_path"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(args.output, index=False)

    validate_balances(train_df, val_df)
    print(f"Saved dataset to {args.output} ({len(combined_df)} rows).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
