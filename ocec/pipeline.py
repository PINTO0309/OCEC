from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import logging
import math
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from PIL import Image
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from .data import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    OCECDataset,
    build_weighted_sampler,
    collect_samples,
    create_dataloader,
    split_samples,
)
from .model import VSDLM, ModelConfig

LOGGER = logging.getLogger("ocec")

LABEL_MAP = {0: "closed", 1: "open"}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEIM_MODULE_PATH = _REPO_ROOT / "03_wholebody34_data_extractor.py"
_DEFAULT_DETECTOR_MODEL = _REPO_ROOT / "deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx"
_DEIM_MODULE: Optional[ModuleType] = None


def _load_deim_module() -> ModuleType:
    global _DEIM_MODULE
    if _DEIM_MODULE is not None:
        return _DEIM_MODULE
    if not _DEIM_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"DEIMv2 demo module not found at {_DEIM_MODULE_PATH}. Please ensure the script is available."
        )
    spec = importlib.util.spec_from_file_location("ocec._deimv2_detector", _DEIM_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load DEIMv2 demo module from {_DEIM_MODULE_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DEIM_MODULE = module
    return module


def _create_mouth_detector(model_path: Path, providers: Sequence[Any]):
    module = _load_deim_module()
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {model_path}")
    provider_list = list(providers) if providers else ["CPUExecutionProvider"]
    try:
        detector = module.DEIMv2(
            runtime="onnx",
            model_path=str(model_path),
            providers=provider_list,
        )
    except Exception as exc:  # pragma: no cover - defensive for runtime issues
        raise RuntimeError(f"Failed to initialize DEIMv2 mouth detector: {exc}") from exc
    return detector


def _resolve_onnx_providers(provider: str) -> List[Any]:
    key = (provider or "cpu").lower()
    if key == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if key == "tensorrt":
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    'trt_engine_cache_enable': True, # .engine, .profile export
                    'trt_engine_cache_path': '.',
                    # 'trt_max_workspace_size': 4e9, # Maximum workspace size for TensorRT engine (1e9 ≈ 1GB)
                    # onnxruntime>=1.21.0 breaking changes
                    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
                    # https://github.com/microsoft/onnxruntime/pull/22681/files
                    # https://github.com/microsoft/onnxruntime/pull/23893/files
                    'trt_op_types_to_exclude': 'NonMaxSuppression,NonZero,RoiAlign',
                    "trt_fp16_enable": True,
                }
            ),
            "CUDAExecutionProvider",
            'CPUExecutionProvider',
        ]
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _extract_mouth_crop(
    frame: np.ndarray,
    box: Any,
    *,
    margin_top: int,
    margin_bottom: int,
    margin_left: int,
    margin_right: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    if frame is None or frame.size == 0 or box is None:
        return None, None
    h, w = frame.shape[:2]
    x1 = max(int(box.x1) - margin_left, 0)
    y1 = max(int(box.y1) - margin_top, 0)
    x2 = min(int(box.x2) + margin_right, w - 1)
    y2 = min(int(box.y2) + margin_bottom, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return None, None
    return crop.copy(), (x1, y1, x2, y2)


if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):

    _grad_scaler_params = inspect.signature(torch.amp.GradScaler).parameters

    if "device" in _grad_scaler_params:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler("cuda", enabled=enabled)

    elif "device_type" in _grad_scaler_params:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler(device_type="cuda", enabled=enabled)

    else:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler(enabled=enabled)

    _autocast_params = inspect.signature(torch.amp.autocast).parameters

    if "device_type" in _autocast_params:

        def _autocast(enabled: bool):
            if not enabled:
                return nullcontext()
            return torch.amp.autocast(device_type="cuda", enabled=True)

    else:

        def _autocast(enabled: bool):
            if not enabled:
                return nullcontext()
            return torch.amp.autocast("cuda", enabled=True)


else:
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    from torch.cuda.amp import autocast as _cuda_autocast

    def _create_grad_scaler(enabled: bool):
        return _CudaGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        return _cuda_autocast(enabled=True)


class RandomCLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), p: float = 0.01) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class _BatchNormAffine(nn.Module):
    def __init__(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        if len(shape) >= 2:
            shape[1] = -1
        return x * self.scale.view(*shape) + self.bias.view(*shape)


def _decompose_batchnorms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            running_mean = child.running_mean.detach()
            running_var = child.running_var.detach()
            if child.affine:
                weight = child.weight.detach()
                bias = child.bias.detach()
            else:
                weight = torch.ones_like(running_mean)
                bias = torch.zeros_like(running_mean)
            scale = weight / torch.sqrt(running_var + child.eps)
            bias_term = bias - running_mean * scale
            affine = _BatchNormAffine(scale, bias_term)
            setattr(module, name, affine)
        else:
            _decompose_batchnorms(child)


def _remove_batchnorm_from_onnx(model):
    from onnx import helper, numpy_helper

    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    value_map = {name: numpy_helper.to_array(init) for name, init in initializer_map.items()}

    additional_initializers = []
    removed_initializers = set()
    new_nodes = []

    for node in graph.node:
        if node.op_type != "BatchNormalization":
            new_nodes.append(node)
            continue

        if len(node.input) < 5:
            new_nodes.append(node)
            continue

        inputs = node.input
        if any(name not in value_map for name in inputs[1:5]):
            new_nodes.append(node)
            continue

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
                break

        scale = value_map[inputs[1]].astype(np.float32)
        bias = value_map[inputs[2]].astype(np.float32)
        mean = value_map[inputs[3]].astype(np.float32)
        var = value_map[inputs[4]].astype(np.float32)

        denom = np.sqrt(var + eps).astype(np.float32)
        alpha = (scale / denom).astype(np.float32)
        beta = (bias - mean * alpha).astype(np.float32)

        alpha_name = f"{node.output[0]}_bn_alpha"
        beta_name = f"{node.output[0]}_bn_beta"
        alpha_init = numpy_helper.from_array(alpha, name=alpha_name)
        beta_init = numpy_helper.from_array(beta, name=beta_name)
        additional_initializers.append(alpha_init)
        additional_initializers.append(beta_init)

        mul_out = f"{node.output[0]}_mul"
        mul_node = helper.make_node(
            "Mul",
            [inputs[0], alpha_name],
            [mul_out],
            name=f"{node.name}_Mul" if node.name else "",
        )
        add_node = helper.make_node(
            "Add",
            [mul_out, beta_name],
            [node.output[0]],
            name=f"{node.name}_Add" if node.name else "",
        )
        new_nodes.extend([mul_node, add_node])
        removed_initializers.update(inputs[1:5])

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    remaining_initializers = [init for init in graph.initializer if init.name not in removed_initializers]
    existing_names = {init.name for init in remaining_initializers}
    for init in additional_initializers:
        if init.name not in existing_names:
            remaining_initializers.append(init)
            existing_names.add(init.name)
    graph.ClearField("initializer")
    graph.initializer.extend(remaining_initializers)

    return model


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    image_size: tuple[int, int] = (112, 112)
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: int = 42
    base_channels: int = 32
    num_blocks: int = 4
    dropout: float = 0.3
    arch_variant: str = "baseline"
    head_variant: str = "auto"
    token_mixer_grid: tuple[int, int] = (2, 3)
    token_mixer_layers: int = 2
    device: str = "auto"
    resume_from: Optional[Path] = None
    use_amp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        data["output_dir"] = str(self.output_dir)
        data["image_size"] = list(self.image_size)
        data["token_mixer_grid"] = list(self.token_mixer_grid)
        if self.resume_from is not None:
            data["resume_from"] = str(self.resume_from)
        return data


def _ensure_image_size_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Image dimensions must be positive integers.")
        return value, value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected tuple of length 2 for image size, got {value!r}.")
        height, width = value
    elif isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"Expected list of length 2 for image size, got {value!r}.")
        height, width = value
    else:
        raise ValueError(f"Unsupported image size specification: {value!r}.")

    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive integers.")
    return height, width


def _parse_image_size_arg(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (tuple, list, int)):
        try:
            return _ensure_image_size_tuple(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    if not isinstance(raw, str):
        raise argparse.ArgumentTypeError(f"Unsupported image size value: {raw!r}")

    text = raw.strip().lower().replace("×", "x").replace(",", "x")
    parts = [part for part in text.split("x") if part]
    try:
        if len(parts) == 1:
            size = int(parts[0])
            return _ensure_image_size_tuple(size)
        if len(parts) == 2:
            height, width = (int(part) for part in parts)
            return _ensure_image_size_tuple((height, width))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

    raise argparse.ArgumentTypeError(
        "Image size must be specified as a single integer (e.g. '48') or as 'HEIGHTxWIDTH' (e.g. '64x48')."
    )


def _parse_token_mixer_grid_arg(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (tuple, list)):
        values = list(raw)
    elif isinstance(raw, str):
        text = raw.strip().lower().replace("×", "x")
        normalized = re.sub(r"[,\s]+", "x", text)
        parts = [part for part in normalized.split("x") if part]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "Token mixer grid must be provided as 'HEIGHTxWIDTH', e.g. '2x3'."
            )
        values = parts
    else:
        raise argparse.ArgumentTypeError(f"Unsupported token mixer grid value: {raw!r}")

    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            "Token mixer grid must contain exactly two positive integers, e.g. '2x3'."
        )

    try:
        height, width = (int(values[0]), int(values[1]))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Token mixer grid values must be integers.") from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Token mixer grid values must be positive integers.")
    return height, width


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec and device_spec.lower() not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device specifier: {device_spec}")
    if device_spec is None or device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_spec)


def _setup_logging(output_dir: Path, verbose: bool) -> None:
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")

    if LOGGER.handlers:
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()

    handlers: List[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    handlers.append(console)

    output_dir.mkdir(parents=True, exist_ok=True)
    logfile = output_dir / "train.log"
    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)

    for handler in handlers:
        LOGGER.addHandler(handler)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_transforms(image_size: Any, mean: Sequence[float], std: Sequence[float]):
    height, width = _ensure_image_size_tuple(image_size)
    train_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=(-20, 20)),
            transforms_v2.RandomPhotometricDistort(p=0.5),
            RandomCLAHE(p=0.01, tile_grid_size=(4, 4)),
            # transforms.RandomGrayscale(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def _compute_pos_weight(samples: Sequence) -> torch.Tensor:
    positives = sum(sample.label for sample in samples)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        LOGGER.warning("Cannot compute class-balanced pos_weight (pos=%d, neg=%d). Using 1.0.", positives, negatives)
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def _infer_accuracy(train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> float:
    if val_metrics is not None:
        acc = val_metrics.get("accuracy")
        if acc is not None and not math.isnan(acc):
            return float(acc)
    return float(train_metrics.get("accuracy", 0.0))


def _prune_checkpoints(directory: Path, prefix: str, max_keep: int) -> None:
    checkpoints = sorted(directory.glob(f"{prefix}*.pt"))
    if len(checkpoints) <= max_keep:
        return
    for path in checkpoints[:-max_keep]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _run_epoch(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any] = None,
    autocast_enabled: bool = False,
    progress_desc: Optional[str] = None,
    collect_outputs: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    if dataloader is None or len(dataloader.dataset) == 0:
        empty_metrics = {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        return empty_metrics, None

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    stats = {"loss": 0.0, "samples": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    collected_probs: List[torch.Tensor] = []
    collected_labels: List[torch.Tensor] = []

    iterator = dataloader
    if progress_desc:
        iterator = tqdm(iterator, desc=progress_desc, leave=False, dynamic_ncols=True)

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with _autocast(autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        stats["loss"] += loss.detach().item() * batch_size
        stats["samples"] += batch_size

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        labels_int = labels.long()
        stats["tp"] += ((preds == 1) & (labels_int == 1)).sum().item()
        stats["tn"] += ((preds == 0) & (labels_int == 0)).sum().item()
        stats["fp"] += ((preds == 1) & (labels_int == 0)).sum().item()
        stats["fn"] += ((preds == 0) & (labels_int == 1)).sum().item()

        if collect_outputs:
            collected_probs.append(probs.detach().cpu())
            collected_labels.append(labels.detach().cpu())

    assert stats["samples"] > 0, "No samples processed during epoch."

    avg_loss = stats["loss"] / stats["samples"]
    accuracy = (stats["tp"] + stats["tn"]) / stats["samples"]
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    extras = None
    if collect_outputs:
        all_probs = torch.cat(collected_probs).squeeze().numpy() if collected_probs else np.array([], dtype=float)
        all_labels = torch.cat(collected_labels).squeeze().numpy() if collected_labels else np.array([], dtype=float)
        extras = {
            "probs": all_probs.astype(float, copy=False),
            "labels": all_labels.astype(int, copy=False),
        }
    return metrics, extras


def _compute_binary_roc_curve(labels: np.ndarray, scores: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    if labels.size == 0 or scores.size == 0:
        return None
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positive_cumsum = np.cumsum(sorted_labels == 1, dtype=float)
    false_positive_cumsum = np.cumsum(sorted_labels == 0, dtype=float)

    tpr = np.concatenate(([0.0], true_positive_cumsum / positives, [1.0]))
    fpr = np.concatenate(([0.0], false_positive_cumsum / negatives, [1.0]))
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _save_epoch_diagnostics(
    labels: np.ndarray,
    scores: np.ndarray,
    split_name: str,
    epoch: int,
    output_dir: Path,
) -> None:
    if labels.size == 0 or scores.size == 0:
        return

    split_dir = output_dir / "diagnostics" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    preds = (scores >= 0.5).astype(int)
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))
    confusion = np.array([[tn, fp], [fn, tp]], dtype=int)

    cm_fig, cm_ax = plt.subplots(figsize=(4, 4))
    cm_im = cm_ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    cm_ax.figure.colorbar(cm_im, ax=cm_ax, fraction=0.046, pad=0.04)
    cm_ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[LABEL_MAP[0], LABEL_MAP[1]],
        yticklabels=[LABEL_MAP[0], LABEL_MAP[1]],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{split_name.capitalize()} Confusion Matrix (epoch {epoch})",
    )
    thresh = confusion.max() / 2 if confusion.max() > 0 else 0.5
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            cm_ax.text(
                j,
                i,
                f"{confusion[i, j]}",
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
            )
    cm_fig.tight_layout()
    cm_path = split_dir / f"confusion_{split_name}_epoch{epoch:04d}.png"
    cm_fig.savefig(cm_path, dpi=150)
    plt.close(cm_fig)

    roc_payload = _compute_binary_roc_curve(labels, scores)
    roc_fig, roc_ax = plt.subplots(figsize=(5, 4))
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    if roc_payload is None:
        roc_ax.set_title(f"{split_name.capitalize()} ROC (epoch {epoch})")
        roc_ax.text(0.5, 0.5, "ROC unavailable (single-class data)", ha="center", va="center")
    else:
        fpr, tpr, auc = roc_payload
        roc_ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        roc_ax.set_title(f"{split_name.capitalize()} ROC (epoch {epoch})")
        roc_ax.legend(loc="lower right")
    roc_fig.tight_layout()
    roc_path = split_dir / f"roc_{split_name}_epoch{epoch:04d}.png"
    roc_fig.savefig(roc_path, dpi=150)
    plt.close(roc_fig)


def _evaluate_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> List[Dict[str, Any]]:
    model.eval()
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            for idx in range(images.size(0)):
                results.append(
                    {
                        "path": batch["path"][idx],
                        "video_name": batch["video_name"][idx],
                        "base_frame": batch["base_frame"][idx],
                        "label": int(batch["label"][idx].item()),
                        "logit": float(logits[idx].detach().cpu().item()),
                        "prob_open": float(probs[idx].detach().cpu().item()),
                    }
                )
    return results


def train_pipeline(config: TrainConfig, verbose: bool = False) -> Dict[str, Any]:
    _setup_logging(config.output_dir, verbose=verbose)
    config_dict = config.to_dict()
    train_config_serialized = copy.deepcopy(config_dict)
    LOGGER.info("Starting training with config: %s", json.dumps(config_dict, indent=2))

    device = _resolve_device(config.device)
    LOGGER.info("Using device: %s", device)
    _set_seed(config.seed)
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    if config.use_amp and not amp_enabled:
        LOGGER.warning("--use_amp requested but CUDA device is not available; proceeding without AMP.")
    tb_dir = config.output_dir
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    history_path = config.output_dir / "history.json"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as fp:
                loaded_history = json.load(fp)
                if isinstance(loaded_history, list):
                    history = loaded_history
        except Exception as exc:
            LOGGER.warning("Failed to load existing history from %s: %s", history_path, exc)
            history = []

    samples = collect_samples(config.data_root, logger=LOGGER)
    splits = split_samples(
        samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        logger=LOGGER,
    )

    mean, std = DEFAULT_MEAN, DEFAULT_STD
    train_transform, eval_transform = _build_transforms(config.image_size, mean, std)
    normalization = {"mean": list(mean), "std": list(std), "image_size": list(config.image_size)}

    train_dataset = OCECDataset(splits["train"], transform=train_transform)
    val_dataset = OCECDataset(splits["val"], transform=eval_transform) if splits["val"] else None
    test_dataset = OCECDataset(splits["test"], transform=eval_transform) if splits["test"] else None

    # train_sampler = build_weighted_sampler(splits["train"])
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=config.num_workers,
    )
    val_loader = (
        create_dataloader(
            val_dataset, batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    test_loader = (
        create_dataloader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
        if test_dataset is not None and len(test_dataset) > 0
        else None
    )

    model_config = ModelConfig(
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        arch_variant=config.arch_variant,
        head_variant=config.head_variant,
        token_mixer_grid=config.token_mixer_grid,
        token_mixer_layers=config.token_mixer_layers,
    )
    model = VSDLM(model_config).to(device)
    base_metadata = {
        "model_config": asdict(model_config),
        "train_config": train_config_serialized,
        "normalization": normalization,
        "label_map": LABEL_MAP,
        "amp_enabled": amp_enabled,
        "tensorboard_logdir": str(tb_dir),
    }

    pos_weight = _compute_pos_weight(splits["train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = _create_grad_scaler(amp_enabled)

    start_epoch = 1
    best_state: Optional[Dict[str, Any]] = None
    best_val_loss = math.inf
    best_f1 = float("-inf")
    best_checkpoint_path: Optional[Path] = None

    if config.resume_from:
        resume_path = config.resume_from
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        resume_epoch = int(resume_payload.get("epoch", 0))
        LOGGER.info("Resuming from checkpoint %s (epoch %d).", resume_path, resume_epoch)

        checkpoint_norm = resume_payload.get("normalization")
        if checkpoint_norm:
            current_size = _ensure_image_size_tuple(normalization["image_size"])
            try:
                checkpoint_size = _ensure_image_size_tuple(checkpoint_norm.get("image_size", current_size))
            except ValueError:
                checkpoint_size = current_size
            checkpoint_mean = list(checkpoint_norm.get("mean", normalization["mean"]))
            checkpoint_std = list(checkpoint_norm.get("std", normalization["std"]))
            norm_mismatch = (
                checkpoint_mean != normalization["mean"]
                or checkpoint_std != normalization["std"]
                or checkpoint_size != current_size
            )
            if norm_mismatch:
                LOGGER.warning(
                    "Checkpoint normalization %s differs from current settings %s.",
                    checkpoint_norm,
                    normalization,
                )

        model.load_state_dict(resume_payload["model_state"])
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state"):
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if resume_payload.get("scaler_state") and amp_enabled:
            scaler.load_state_dict(resume_payload["scaler_state"])

        start_epoch = resume_epoch + 1
        resume_train_metrics = copy.deepcopy(resume_payload.get("train_metrics") or {})
        resume_val_metrics = copy.deepcopy(resume_payload.get("val_metrics") or None)
        best_val_loss = resume_payload.get("best_val_loss")
        if best_val_loss is None:
            best_val_loss = (
                resume_val_metrics["loss"]
                if isinstance(resume_val_metrics, dict) and "loss" in resume_val_metrics
                else math.inf
            )
        best_accuracy = resume_payload.get("best_accuracy")
        if best_accuracy is None:
            best_accuracy = _infer_accuracy(
                resume_train_metrics,
                resume_val_metrics if isinstance(resume_val_metrics, dict) else None,
            )
        best_f1_candidate = resume_payload.get("best_f1")
        if best_f1_candidate is not None:
            best_f1 = float(best_f1_candidate)
        else:
            candidate_sources = [
                resume_val_metrics if isinstance(resume_val_metrics, dict) else None,
                resume_train_metrics if isinstance(resume_train_metrics, dict) else None,
            ]
            extracted_f1 = None
            for source in candidate_sources:
                if source and source.get("f1") is not None and not math.isnan(source.get("f1")):
                    extracted_f1 = float(source["f1"])
                    break
            if extracted_f1 is not None:
                best_f1 = extracted_f1
            else:
                best_f1 = float("-inf")

        best_state = {
            "epoch": resume_payload.get("best_epoch", resume_epoch),
            "model_state": copy.deepcopy(resume_payload["model_state"]),
            "optimizer_state": copy.deepcopy(resume_payload.get("optimizer_state")),
            "scheduler_state": copy.deepcopy(resume_payload.get("scheduler_state")),
            "scaler_state": copy.deepcopy(resume_payload.get("scaler_state")),
            "train_metrics": resume_train_metrics,
            "val_metrics": resume_val_metrics,
            "best_val_loss": best_val_loss,
            "best_accuracy": best_accuracy,
            "best_f1": best_f1,
            "checkpoint_path": str(resume_path),
        }
        best_checkpoint_path = resume_path

        if history:
            history = [entry for entry in history if entry.get("epoch", 0) < start_epoch]

        if start_epoch > config.epochs:
            LOGGER.info(
                "Checkpoint epoch %d exceeds requested total epochs %d; no additional training will be performed.",
                start_epoch - 1,
                config.epochs,
            )

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics, train_outputs = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            autocast_enabled=amp_enabled,
            progress_desc=f"Train {epoch}/{config.epochs}",
            collect_outputs=True,
        )
        if val_loader:
            val_metrics, val_outputs = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                autocast_enabled=amp_enabled,
                progress_desc=f"Val   {epoch}/{config.epochs}",
                collect_outputs=True,
            )
        else:
            val_metrics, val_outputs = None, None

        if val_metrics:
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step(train_metrics["loss"])

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics or {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        train_summary = (
            f"loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} "
            f"prec {train_metrics['precision']:.4f} "
            f"rec {train_metrics['recall']:.4f} "
            f"f1 {train_metrics['f1']:.4f}"
        )
        if val_metrics:
            val_summary = (
                f"loss {val_metrics['loss']:.4f} "
                f"acc {val_metrics['accuracy']:.4f} "
                f"prec {val_metrics['precision']:.4f} "
                f"rec {val_metrics['recall']:.4f} "
                f"f1 {val_metrics['f1']:.4f}"
            )
        else:
            val_summary = "loss n/a acc n/a prec n/a rec n/a f1 n/a"

        LOGGER.info("Epoch %d | train %s | val %s", epoch, train_summary, val_summary)

        tb_writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        tb_writer.add_scalar("metrics/train_accuracy", train_metrics["accuracy"], epoch)
        tb_writer.add_scalar("metrics/train_f1", train_metrics["f1"], epoch)
        if val_metrics is not None:
            tb_writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            tb_writer.add_scalar("metrics/val_accuracy", val_metrics["accuracy"], epoch)
            tb_writer.add_scalar("metrics/val_f1", val_metrics["f1"], epoch)
        tb_writer.flush()

        if train_outputs is not None:
            _save_epoch_diagnostics(train_outputs["labels"], train_outputs["probs"], "train", epoch, config.output_dir)
        if val_outputs is not None:
            _save_epoch_diagnostics(val_outputs["labels"], val_outputs["probs"], "val", epoch, config.output_dir)
        train_outputs = None
        val_outputs = None

        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        epoch_payload = {
            **base_metadata,
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        epoch_path = config.output_dir / f"vsdlm_epoch_{epoch:04d}.pt"
        torch.save(epoch_payload, epoch_path)
        _prune_checkpoints(config.output_dir, "vsdlm_epoch_", 10)

        current_val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
        score_value = val_metrics["f1"] if val_metrics else train_metrics["f1"]
        if score_value > best_f1:
            accuracy_value = _infer_accuracy(train_metrics, val_metrics)
            best_f1 = score_value
            best_val_loss = current_val_loss
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model_state),
                "optimizer_state": copy.deepcopy(optimizer_state),
                "scheduler_state": copy.deepcopy(scheduler_state),
                "scaler_state": copy.deepcopy(scaler.state_dict()) if amp_enabled else None,
                "train_metrics": copy.deepcopy(train_metrics),
                "val_metrics": copy.deepcopy(val_metrics) if val_metrics is not None else None,
                "best_val_loss": current_val_loss,
                "best_accuracy": accuracy_value,
                "best_f1": score_value,
            }
            best_checkpoint = dict(epoch_payload)
            best_checkpoint.update(
                best_val_loss=current_val_loss,
                best_epoch=epoch,
                best_accuracy=accuracy_value,
                best_f1=score_value,
            )
            best_checkpoint_path = config.output_dir / f"vsdlm_best_epoch{epoch:04d}_f1_{score_value:.4f}.pt"
            best_state["checkpoint_path"] = str(best_checkpoint_path)
            torch.save(best_checkpoint, best_checkpoint_path)
            _prune_checkpoints(config.output_dir, "vsdlm_best_", 10)
            LOGGER.info("New best model at epoch %d (F1 %.4f).", epoch, score_value)

    if best_state is None or best_checkpoint_path is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state["model_state"])

    test_metrics = None
    if test_loader:
        test_metrics, _ = _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            autocast_enabled=amp_enabled,
            progress_desc="Test",
        )
    LOGGER.info("Test metrics: %s", json.dumps(test_metrics, indent=2) if test_metrics else "n/a")
    if test_metrics:
        step = best_state["epoch"]
        tb_writer.add_scalar("loss/test", test_metrics["loss"], step)
        tb_writer.add_scalar("metrics/test_accuracy", test_metrics["accuracy"], step)
        tb_writer.add_scalar("metrics/test_f1", test_metrics["f1"], step)
        tb_writer.flush()

    with open(config.output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    if test_loader:
        predictions = _evaluate_predictions(model, test_loader, device)
        pd.DataFrame(predictions).to_csv(config.output_dir / "test_predictions.csv", index=False)

    summary = {
        "checkpoint": str(best_checkpoint_path),
        "best_epoch": best_state["epoch"],
        "best_accuracy": best_state["best_accuracy"],
        "best_f1": best_state["best_f1"],
        "best_val_loss": best_state["best_val_loss"],
        "val_metrics": best_state["val_metrics"],
        "train_metrics": best_state["train_metrics"],
        "amp_enabled": amp_enabled,
        "resume_from": str(config.resume_from) if config.resume_from else None,
        "history_path": str(config.output_dir / "history.json"),
        "test_metrics": test_metrics,
        "retained_checkpoints": {
            "epoch": 10,
            "best": 10,
        },
        "tensorboard_logdir": str(tb_dir),
    }
    with open(config.output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    tb_writer.close()
    return summary


def _gather_image_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.glob("**/*.png") if p.is_file()))
        else:
            LOGGER.warning("Skipping missing path: %s", item)
    if not paths:
        raise FileNotFoundError("No images found for inference.")
    return paths


def predict_images(
    checkpoint_path: Path,
    inputs: Sequence[str],
    device_spec: str = "auto",
) -> pd.DataFrame:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = VSDLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    image_size_raw = normalization.get("image_size", (112, 112))
    try:
        image_size = _ensure_image_size_tuple(image_size_raw)
    except ValueError:
        LOGGER.warning("Invalid image_size %s in checkpoint; defaulting to 112x112.", image_size_raw)
        image_size = (112, 112)
    _, eval_transform = _build_transforms(image_size, mean, std)

    image_paths = _gather_image_paths(inputs)

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert("RGB")
            tensor = eval_transform(pil_image).unsqueeze(0).to(device)
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0].item()
            records.append(
                {
                    "path": str(image_path),
                    "logit": float(logits.item()),
                    "prob_open": float(prob),
                }
            )

    return pd.DataFrame(records)


def run_webcam_inference(
    checkpoint_path: Path,
    camera_index: int = 0,
    device_spec: str = "auto",
    window_name: str = "OCEC Webcam",
    mirror: bool = False,
    detector_model: Optional[Path] = None,
    detector_provider: str = "cpu",
) -> None:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = VSDLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    image_size_raw = normalization.get("image_size", (112, 112))
    try:
        image_size = _ensure_image_size_tuple(image_size_raw)
    except ValueError:
        LOGGER.warning("Invalid image_size %s in checkpoint; defaulting to 112x112.", image_size_raw)
        image_size = (112, 112)
    _, eval_transform = _build_transforms(image_size, mean, std)

    detector_path = Path(detector_model) if detector_model is not None else _DEFAULT_DETECTOR_MODEL
    if not detector_path.exists():
        raise FileNotFoundError(
            f"Mouth detector model not found at {detector_path}. "
            "Provide --detector_model pointing to a valid DEIMv2 ONNX file."
        )
    detector_providers = _resolve_onnx_providers(detector_provider)
    mouth_detector = _create_mouth_detector(detector_path, detector_providers)
    LOGGER.info("Loaded DEIMv2 mouth detector from %s using providers %s.", detector_path, detector_providers)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}.")

    LOGGER.info(
        "Starting webcam inference using checkpoint %s on device %s (camera index %d).",
        checkpoint_path,
        device,
        camera_index,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    margin_top = 2
    margin_bottom = 6
    margin_left = 2
    margin_right = 2

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    LOGGER.warning("Failed to read frame from camera; stopping.")
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                try:
                    boxes = mouth_detector(
                        image=frame,
                        disable_generation_identification_mode=True,
                        disable_gender_identification_mode=True,
                        disable_left_and_right_hand_identification_mode=True,
                        disable_headpose_identification_mode=True,
                    )
                except Exception as exc:  # pragma: no cover - runtime safeguard
                    LOGGER.error("Mouth detector inference failed: %s", exc)
                    break

                mouth_boxes = [box for box in boxes if getattr(box, "classid", None) == 19]
                best_box = max(mouth_boxes, key=lambda box: box.score) if mouth_boxes else None
                crop, crop_coords = _extract_mouth_crop(
                    frame,
                    best_box,
                    margin_top=margin_top,
                    margin_bottom=margin_bottom,
                    margin_left=margin_left,
                    margin_right=margin_right,
                )

                if crop is None:
                    cv2.putText(
                        frame,
                        "Mouth not detected",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        LOGGER.info("Exit requested (key press).")
                        break
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(rgb_crop)
                tensor = eval_transform(pil_crop).unsqueeze(0).to(device)
                logits = model(tensor)
                prob_open = torch.sigmoid(logits)[0].item()

                label = LABEL_MAP[int(prob_open >= 0.5)]
                color = (0, 200, 0) if label == "open" else (50, 50, 255)
                cv2.putText(
                    frame,
                    f"Prob open: {prob_open:.2%}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Prediction: {label}",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if crop_coords is not None:
                    x1, y1, x2, y2 = crop_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    LOGGER.info("Exit requested (key press).")
                    break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def run_webcam_inference_onnx(
    onnx_model: Path,
    camera_index: int = 0,
    window_name: str = "OCEC Webcam (ONNX)",
    mirror: bool = False,
    model_provider: str = "cpu",
    detector_model: Optional[Path] = None,
    detector_provider: str = "cpu",
    image_size: Optional[tuple[int, int]] = None,
) -> None:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("onnxruntime is required for ONNX inference. Install with `pip install onnxruntime`.") from exc

    ort.set_default_logger_severity(3)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    providers = _resolve_onnx_providers(model_provider)
    session = ort.InferenceSession(
        str(onnx_model),
        sess_options=session_options,
        providers=providers,
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_shape = session.get_inputs()[0].shape
    if image_size is not None:
        height, width = _ensure_image_size_tuple(image_size)
        if len(input_shape) >= 4:
            expected_h, expected_w = input_shape[2], input_shape[3]
            if isinstance(expected_h, int) and isinstance(expected_w, int):
                if (expected_h, expected_w) != (height, width):
                    raise ValueError(
                        f"Specified image_size {(height, width)} does not match ONNX input {(expected_h, expected_w)}."
                    )
    else:
        if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            height = int(input_shape[2])
            width = int(input_shape[3])
        else:
            LOGGER.warning(
                "ONNX model has dynamic spatial dimensions; defaulting to 112x112. "
                "Override with --image_size HEIGHTxWIDTH if needed."
            )
            height, width = (112, 112)
    _, eval_transform = _build_transforms((height, width), DEFAULT_MEAN, DEFAULT_STD)

    detector_path = Path(detector_model) if detector_model is not None else _DEFAULT_DETECTOR_MODEL
    if not detector_path.exists():
        raise FileNotFoundError(
            f"Mouth detector model not found at {detector_path}. "
            "Provide --detector_model pointing to a valid DEIMv2 ONNX file."
        )
    detector_providers = _resolve_onnx_providers(detector_provider)
    mouth_detector = _create_mouth_detector(detector_path, detector_providers)
    LOGGER.info(
        "Loaded ONNX model %s with providers %s (detector providers %s).",
        onnx_model,
        providers,
        detector_providers,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}.")

    LOGGER.info(
        "Starting ONNX webcam inference using %s (camera index %d).",
        onnx_model,
        camera_index,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    margin_top = 5 #2
    margin_bottom = 5 #6
    margin_left = 5 #2
    margin_right = 5 #2

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                LOGGER.warning("Failed to read frame from camera; stopping.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            try:
                boxes = mouth_detector(
                    image=frame,
                    disable_generation_identification_mode=True,
                    disable_gender_identification_mode=True,
                    disable_left_and_right_hand_identification_mode=True,
                    disable_headpose_identification_mode=True,
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOGGER.error("Mouth detector inference failed: %s", exc)
                break

            mouth_boxes = [box for box in boxes if getattr(box, "classid", None) == 19]
            best_box = max(mouth_boxes, key=lambda box: box.score) if mouth_boxes else None
            crop, crop_coords = _extract_mouth_crop(
                frame,
                best_box,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right,
            )

            if crop is None:
                cv2.putText(
                    frame,
                    "Mouth not detected",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    LOGGER.info("Exit requested (key press).")
                    break
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(rgb_crop)
            tensor = eval_transform(pil_crop).unsqueeze(0)
            inputs = {input_name: tensor.cpu().numpy()}
            outputs = session.run([output_name], inputs)
            prob_open = float(outputs[0].flatten()[0])

            label = LABEL_MAP[int(prob_open >= 0.5)]
            color = (0, 200, 0) if label == "open" else (50, 50, 255)
            cv2.putText(
                frame,
                f"Prob open: {prob_open:.2%}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            if crop_coords is not None:
                x1, y1, x2, y2 = crop_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                LOGGER.info("Exit requested (key press).")
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    device_spec: str = "auto",
) -> None:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = VSDLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    image_size_raw = normalization.get("image_size", (112, 112))
    try:
        image_size = _ensure_image_size_tuple(image_size_raw)
    except ValueError:
        LOGGER.warning("Invalid image_size %s in checkpoint; defaulting to 112x112.", image_size_raw)
        image_size = (112, 112)

    dummy = torch.randn(1, 3, image_size[0], image_size[1], device=device)

    class _ONNXProbWrapper(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.base_model(x)
            return torch.sigmoid(logits)

    export_base = copy.deepcopy(model)
    export_base.to(device)
    export_base.eval()
    export_model = _ONNXProbWrapper(export_base)

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["prob_open"],
        dynamic_axes={"images": {0: "batch"}, "prob_open": {0: "batch"}},
        opset_version=opset,
    )
    LOGGER.info("Exported ONNX model to %s", output_path)

    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(output_path)
        simplified_model, check = simplify(onnx_model)
        if check:
            simplified_model = _remove_batchnorm_from_onnx(simplified_model)
            onnx.save(simplified_model, output_path)
            LOGGER.info("Simplified ONNX model with onnxsim at %s", output_path)
        else:
            LOGGER.warning("onnxsim simplification check failed; keeping original model.")
    except Exception as exc:
        LOGGER.warning("onnxsim simplification failed: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCEC training and inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the OCEC binary classifier.")
    train_parser.add_argument("--data_root", type=Path, required=True)
    train_parser.add_argument("--output_dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument(
        "--image_size",
        type=_parse_image_size_arg,
        default=_parse_image_size_arg("48"),
        help="Square size (e.g. 48) or HEIGHTxWIDTH (e.g. 64x48) for resizing input images.",
    )
    train_parser.add_argument("--train_ratio", type=float, default=0.8)
    train_parser.add_argument("--val_ratio", type=float, default=0.2)
    train_parser.add_argument("--test_ratio", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--base_channels", type=int, default=32)
    train_parser.add_argument("--num_blocks", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument(
        "--arch_variant",
        type=str,
        default="baseline",
        choices=["baseline", "inverted_se", "convnext"],
        help="Backbone architecture choice.",
    )
    train_parser.add_argument(
        "--head_variant",
        type=str,
        default="auto",
        choices=["auto", "avg", "avgmax_mlp", "transformer", "mlp_mixer"],
        help="Classification head configuration.",
    )
    train_parser.add_argument(
        "--token_mixer_grid",
        type=_parse_token_mixer_grid_arg,
        default=(2, 3),
        help="Grid for token mixer heads specified as HEIGHTxWIDTH (e.g. '2x3').",
    )
    train_parser.add_argument(
        "--token_mixer_layers",
        type=int,
        default=2,
        help="Number of transformer or MLP-mixer layers when using those head variants.",
    )
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--resume", type=Path, help="Resume training from a checkpoint file.")
    train_parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training (CUDA only).")
    train_parser.add_argument("--verbose", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained checkpoint.")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--inputs", nargs="+", required=True, help="Image files or directories.")
    predict_parser.add_argument("--output", type=Path, help="Optional CSV path to save predictions.")
    predict_parser.add_argument("--device", type=str, default="auto")

    webcam_parser = subparsers.add_parser("webcam", help="Run real-time inference from a webcam.")
    webcam_parser.add_argument("--checkpoint", type=Path, required=True)
    webcam_parser.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index (default: 0).")
    webcam_parser.add_argument("--device", type=str, default="auto")
    webcam_parser.add_argument("--window_name", type=str, default="OCEC Webcam")
    webcam_parser.add_argument("--mirror", action="store_true", help="Mirror frames horizontally before display.")
    webcam_parser.add_argument(
        "--detector_model",
        type=Path,
        help="Optional path to the DEIMv2 ONNX model for mouth detection. Defaults to the bundled dataset weight.",
    )
    webcam_parser.add_argument(
        "--detector_provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
        help="ONNX Runtime execution provider for the mouth detector (default: cpu).",
    )

    webcam_onnx_parser = subparsers.add_parser("webcam_onnx", help="Run real-time inference from a webcam using an ONNX model.")
    webcam_onnx_parser.add_argument("--model", type=Path, required=True, help="Path to the exported OCEC ONNX model.")
    webcam_onnx_parser.add_argument("--camera_index", type=int, default=0, help="OpenCV camera index (default: 0).")
    webcam_onnx_parser.add_argument("--window_name", type=str, default="OCEC Webcam (ONNX)")
    webcam_onnx_parser.add_argument("--mirror", action="store_true", help="Mirror frames horizontally before display.")
    webcam_onnx_parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
        help="ONNX Runtime execution provider for the classifier (default: cpu).",
    )
    webcam_onnx_parser.add_argument(
        "--detector_model",
        type=Path,
        help="Optional path to the DEIMv2 ONNX model for mouth detection. Defaults to the bundled dataset weight.",
    )
    webcam_onnx_parser.add_argument(
        "--detector_provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "tensorrt"],
        help="ONNX Runtime execution provider for the mouth detector (default: cpu).",
    )
    webcam_onnx_parser.add_argument(
        "--image_size",
        type=str,
        default=None,
        help="Override classifier input size as HEIGHTxWIDTH (defaults to ONNX model shape).",
    )

    onnx_parser = subparsers.add_parser("exportonnx", help="Export a trained checkpoint to ONNX.")
    onnx_parser.add_argument("--checkpoint", type=Path, required=True)
    onnx_parser.add_argument("--output", type=Path, required=True)
    onnx_parser.add_argument("--opset", type=int, default=17)
    onnx_parser.add_argument("--device", type=str, default="auto")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            image_size=args.image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            arch_variant=args.arch_variant,
            head_variant=args.head_variant,
            token_mixer_grid=args.token_mixer_grid,
            token_mixer_layers=args.token_mixer_layers,
            device=args.device,
            resume_from=args.resume,
            use_amp=args.use_amp,
        )
        train_pipeline(config, verbose=args.verbose)
    elif args.command == "predict":
        df = predict_images(args.checkpoint, args.inputs, device_spec=args.device)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved predictions to {args.output}")
        else:
            print(df.to_string(index=False))
    elif args.command == "webcam":
        run_webcam_inference(
            args.checkpoint,
            camera_index=args.camera_index,
            device_spec=args.device,
            window_name=args.window_name,
            mirror=args.mirror,
            detector_model=args.detector_model,
            detector_provider=args.detector_provider,
        )
    elif args.command == "webcam_onnx":
        parsed_size = _parse_image_size_arg(args.image_size) if args.image_size else None
        run_webcam_inference_onnx(
            args.model,
            camera_index=args.camera_index,
            window_name=args.window_name,
            mirror=args.mirror,
            model_provider=args.provider,
            detector_model=args.detector_model,
            detector_provider=args.detector_provider,
            image_size=parsed_size,
        )
    elif args.command == "exportonnx":
        export_to_onnx(args.checkpoint, args.output, opset=args.opset, device_spec=args.device)
    else:
        parser.error(f"Unknown command: {args.command}")
