from __future__ import annotations

import io
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

DEFAULT_MEAN = [0.0, 0.0, 0.0]
DEFAULT_STD = [1.0, 1.0, 1.0]

_SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    """Lightweight container describing a single cropped eye frame."""

    index: int
    path: str
    resolved_path: Optional[Path]
    label: int
    label_name: str
    split: str
    source: str
    base_frame: str
    video_name: str
    image_bytes: Optional[bytes]


def _resolve_dataset_path(data_root: Path) -> Path:
    if data_root.is_file() and data_root.suffix == ".parquet":
        return data_root
    candidate = data_root / "dataset.parquet"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate dataset parquet under {data_root}.")


def _normalize_split(value: str) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _SPLIT_ALIASES.get(key)


def _to_label(row: pd.Series) -> int:
    if "class_id" in row and pd.notna(row["class_id"]):
        return int(row["class_id"])
    label_text = str(row.get("label", "")).strip().lower()
    if label_text not in {"open", "closed"}:
        raise ValueError(f"Unsupported label value: {label_text!r}")
    return 1 if label_text == "open" else 0


def _prepare_image_bytes(value: object) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


def collect_samples(data_root: Path, logger: Optional[logging.Logger] = None) -> List[Sample]:
    """Load samples from the parquet dataset."""

    logger = logger or logging.getLogger(__name__)
    dataset_path = _resolve_dataset_path(data_root)
    df = pd.read_parquet(dataset_path)
    if "split" not in df.columns:
        raise ValueError(f"Dataset at {dataset_path} must contain a 'split' column.")

    samples: List[Sample] = []
    dataset_dir = dataset_path.parent
    for idx, row in df.iterrows():
        split = _normalize_split(row.get("split"))
        if split is None:
            logger.debug("Skipping row %d with unsupported split value %r.", idx, row.get("split"))
            continue

        raw_path = str(row.get("image_path", "")).strip()
        if not raw_path:
            logger.debug("Skipping row %d without image_path.", idx)
            continue

        display_path = raw_path.replace("\\", "/")
        path_candidate = Path(display_path)
        resolved_path: Optional[Path] = None
        if path_candidate.is_absolute() and path_candidate.exists():
            resolved_path = path_candidate
        else:
            candidate = (dataset_dir / path_candidate).resolve()
            if candidate.exists():
                resolved_path = candidate

        label = _to_label(row)
        label_name = str(row.get("label", "open" if label == 1 else "closed"))
        source = str(row.get("source", "unknown") or "unknown")
        base_frame = Path(display_path).name
        video_group = source if source else Path(display_path).parent.name

        samples.append(
            Sample(
                index=int(idx),
                path=display_path,
                resolved_path=resolved_path,
                label=label,
                label_name=label_name,
                split=split,
                source=source,
                base_frame=base_frame,
                video_name=video_group,
                image_bytes=_prepare_image_bytes(row.get("image_bytes")),
            )
        )

    if not samples:
        raise RuntimeError(f"No usable samples found in {dataset_path}.")

    total = len(samples)
    counts = Counter(sample.split for sample in samples)
    logger.info(
        "Loaded %d samples from %s (splits: %s).",
        total,
        dataset_path,
        ", ".join(f"{name}={counts.get(name, 0)}" for name in ("train", "val", "test")),
    )
    return samples


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,  # pragma: no cover - retained for interface compatibility
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Sample]]:
    """Group samples by pre-defined split labels."""

    logger = logger or logging.getLogger(__name__)
    logger.info(
        "Using pre-defined dataset splits; ratio arguments (train=%.3f, val=%.3f, test=%.3f) are informational only.",
        train_ratio,
        val_ratio,
        test_ratio,
    )
    assignments: Dict[str, List[Sample]] = {split: [] for split in ("train", "val", "test")}
    for sample in samples:
        if sample.split in assignments:
            assignments[sample.split].append(sample)
        else:
            logger.debug("Ignoring sample %s with unsupported split '%s'.", sample.path, sample.split)

    for split, subset in assignments.items():
        if not subset:
            continue
        positives = sum(sample.label for sample in subset)
        total = len(subset)
        open_ratio = (positives / total) * 100.0 if total else 0.0
        logger.info(
            "Split '%s': %d samples from %d sources (%.1f%% labelled 'open').",
            split,
            total,
            len({sample.video_name for sample in subset}),
            open_ratio,
        )

    if not assignments["train"]:
        raise ValueError("Training split is empty. Ensure the dataset includes 'train' rows.")

    return assignments


class OCECDataset(Dataset):
    """Torch dataset that reads eyes crops from disk or embedded bytes."""

    def __init__(self, samples: Sequence[Sample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: Sample) -> Image.Image:
        if sample.image_bytes is not None:
            return Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
        if sample.resolved_path is None:
            raise FileNotFoundError(f"Image file not found for sample {sample.path!r}.")
        return Image.open(sample.resolved_path).convert("RGB")

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "video_name": sample.video_name,
            "path": sample.path,
            "base_frame": sample.base_frame,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 8,
    sampler: Optional[WeightedRandomSampler] = None,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_weighted_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler:
    labels = [sample.label for sample in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
