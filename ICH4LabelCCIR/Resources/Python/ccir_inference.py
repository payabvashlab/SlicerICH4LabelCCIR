#!/usr/bin/env python3
"""
Deployment inference for ORBIT-RSNA four-label segmentation plus CCIR refinement.

Pipeline
--------
1. Run the trained method2 ORBIT-RSNA nnU-Net and obtain labels:
   0 background, 1 ICH, 2 edema, 3 ventricle, 4 IVH.
2. Find 26-connected predicted-IVH components.
3. Send only components smaller than the physical-volume gate (default 1 mL)
   through the trained 2.5-D global + local 3-D CCIR network.
4. Remove a component to background when P(true IVH) is below the calibrated
   removal threshold; preserve uncertain components; optionally add IVH voxels
   near a high-confidence component using the residual head.
5. Preserve method2 labels 1, 2 and 3 everywhere.

This file intentionally contains no Slicer imports, so it can run in an isolated
PythonSlicer process without blocking the Slicer user interface.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable


def configure_gpu_before_torch_import(argv: list[str]) -> None:
    gpu = "0"
    if "--gpu" in argv:
        index = argv.index("--gpu")
        if index + 1 < len(argv):
            gpu = argv[index + 1]
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


configure_gpu_before_torch_import(sys.argv)

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from nibabel.processing import resample_from_to
from scipy.ndimage import (
    binary_dilation,
    center_of_mass,
    distance_transform_edt,
    generate_binary_structure,
    label as connected_components,
    zoom,
)
from torch import nn


LABEL_NAMES = {0: "background", 1: "ICH", 2: "Edema", 3: "Ventricle", 4: "IVH"}
IVH_LABEL = 4
VENTRICLE_LABEL = 3
FEATURE_COUNT = 12
DEFAULT_GLOBAL_OFFSETS = (-2, -1, 0, 1, 2)
DEFAULT_WINDOWS_HU = ((-20.0, 100.0), (0.0, 80.0), (40.0, 140.0))
DEFAULT_GLOBAL_HW = (256, 256)
DEFAULT_LOCAL_DHW = (5, 96, 96)
DEFAULT_RESIDUAL_VOXEL_THRESHOLD = 0.50
DEFAULT_RESIDUAL_DILATION_ITERATIONS = 3
EXPECTED_ORBIT_TRAINER = "nnUNetTrainerORBITRSNA"
EXPECTED_CCIR_ARCHITECTURE = "CCIR_DenseNet121_2p5D_Local3D_v1"


def log(message: str) -> None:
    print(message, flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One *_0000 NIfTI CT")
    inputs.add_argument("--input-folder", type=Path, help="Recursive NIfTI batch folder")
    parser.add_argument("--output", type=Path, help="Single-case segmentation path")
    parser.add_argument("--report", type=Path, help="Single-case JSON report path")
    parser.add_argument("--output-folder", type=Path, help="Batch output folder")
    parser.add_argument("--case-id", help="Output case ID for single-volume mode")
    parser.add_argument("--nnunet-model-dir", type=Path, required=True)
    parser.add_argument("--ccir-checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainerORBITRSNA")
    parser.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--nnunet-checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--component-volume-threshold-ml", type=float, default=1.0)
    parser.add_argument("--remove-threshold", type=float, default=0.5)
    parser.add_argument("--keep-refine-threshold", type=float, default=0.9)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--npp", type=int, default=1)
    parser.add_argument("--nps", type=int, default=1)
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--disable-residual-addition", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input is not None and (args.output is None or args.report is None):
        parser.error("single mode requires --output and --report")
    if args.input_folder is not None and args.output_folder is None:
        parser.error("batch mode requires --output-folder")
    if args.component_volume_threshold_ml <= 0:
        parser.error("--component-volume-threshold-ml must be > 0")
    if not 0 <= args.remove_threshold <= 1:
        parser.error("--remove-threshold must be in [0, 1]")
    if not 0 <= args.keep_refine_threshold <= 1:
        parser.error("--keep-refine-threshold must be in [0, 1]")
    if args.keep_refine_threshold < args.remove_threshold:
        parser.error("keep/refine threshold must be >= remove threshold")
    return args


def verify_runtime_dependencies() -> None:
    """Fail before nnU-Net if the Stage-2 runtime is incomplete."""
    required = {
        "torch": "Slicer PyTorch extension",
        "nnunetv2": "nnunetv2==2.8.1",
        "monai": "monai>=1.3,<2",
        "nibabel": "nibabel>=5.1",
        "scipy": "scipy>=1.10",
    }
    missing = [
        module for module in required if importlib.util.find_spec(module) is None
    ]
    if missing:
        instructions = ", ".join(
            f"{module} ({required[module]})" for module in missing
        )
        raise RuntimeError(
            "Missing Python dependencies: "
            + instructions
            + ". Install/verify dependencies in the Slicer module and restart "
            "Slicer before running inference."
        )
    import monai

    log(f"Dependency preflight: MONAI {monai.__version__}, PyTorch {torch.__version__}")


def install_bundled_orbit_trainer(trainer_name: str) -> Path:
    """Install the inference-only ORBIT architecture into Slicer's nnU-Net."""
    if trainer_name != EXPECTED_ORBIT_TRAINER:
        raise ValueError(
            f"This extension requires trainer {EXPECTED_ORBIT_TRAINER}, got "
            f"{trainer_name}. Select the method2 ORBIT-RSNA results folder."
        )
    import nnunetv2

    source = Path(__file__).resolve().with_name(f"{EXPECTED_ORBIT_TRAINER}.py")
    if not source.is_file():
        raise FileNotFoundError(f"Bundled ORBIT trainer is missing: {source}")
    destination = (
        Path(nnunetv2.__file__).resolve().parent
        / "training"
        / "nnUNetTrainer"
        / source.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or source.read_bytes() != destination.read_bytes():
        shutil.copy2(source, destination)
        importlib.invalidate_caches()
        log(f"Installed bundled ORBIT inference trainer: {destination}")
    else:
        log(f"Bundled ORBIT inference trainer verified: {destination}")
    return destination


def resolve_nnunet_layout(
    selected: Path,
    trainer: str,
    plans: str,
    configuration: str,
) -> tuple[Path, Path]:
    """Accept either a Dataset folder or trainer__plans__configuration folder."""
    selected = selected.expanduser().resolve()
    expected_configuration_name = f"{trainer}__{plans}__{configuration}"
    if selected.name == expected_configuration_name or (
        "__" in selected.name and (selected / "dataset.json").is_file()
    ):
        configuration_directory = selected
        dataset_directory = selected.parent
    else:
        dataset_directory = selected
        configuration_directory = dataset_directory / expected_configuration_name
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"nnU-Net Dataset folder not found: {dataset_directory}")
    if not configuration_directory.is_dir():
        raise FileNotFoundError(
            f"nnU-Net configuration folder not found: {configuration_directory}"
        )
    dataset_json_candidates = (
        configuration_directory / "dataset.json",
        dataset_directory / "dataset.json",
    )
    if not any(path.is_file() for path in dataset_json_candidates):
        raise FileNotFoundError(
            "dataset.json is missing from both the configuration and Dataset folders"
        )
    return dataset_directory, configuration_directory


def parse_folds(text: str) -> list[str]:
    values = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    return values or ["all"]


def validate_nnunet_weights(
    configuration_directory: Path,
    folds: Iterable[str],
    checkpoint_name: str,
) -> None:
    missing: list[str] = []
    for fold in folds:
        fold_name = "fold_all" if str(fold).lower() == "all" else f"fold_{fold}"
        checkpoint = configuration_directory / fold_name / checkpoint_name
        if not checkpoint.is_file():
            missing.append(str(checkpoint))
    if missing:
        raise FileNotFoundError("Missing nnU-Net checkpoint(s):\n" + "\n".join(missing))


def find_nnunet_predict() -> Path:
    names = ("nnUNetv2_predict.exe", "nnUNetv2_predict")
    on_path = shutil.which("nnUNetv2_predict") or shutil.which("nnUNetv2_predict.exe")
    if on_path:
        return Path(on_path)
    executable = Path(sys.executable).resolve()
    roots = [
        executable.parent,
        executable.parent / "Scripts",
        executable.parent / "bin",
        executable.parent.parent / "lib" / "Python" / "Scripts",
        executable.parent.parent / "lib" / "Python" / "bin",
        executable.parent.parent / "bin",
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "nnUNetv2_predict was not found in Slicer's Python environment. "
        "Use the extension's dependency installer and restart Slicer."
    )


def run_nnunet(
    args: argparse.Namespace,
    input_directory: Path,
    baseline_directory: Path,
) -> None:
    dataset_directory, configuration_directory = resolve_nnunet_layout(
        args.nnunet_model_dir, args.trainer, args.plans, args.configuration
    )
    folds = parse_folds(args.folds)
    validate_nnunet_weights(configuration_directory, folds, args.nnunet_checkpoint)
    predict = find_nnunet_predict()
    baseline_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["nnUNet_results"] = str(dataset_directory.parent)
    environment.setdefault("nnUNet_raw", str(dataset_directory.parent))
    environment.setdefault("nnUNet_preprocessed", str(dataset_directory.parent))
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_cuda = bool(torch.cuda.is_available())
    command = [
        str(predict),
        "-i", str(input_directory),
        "-o", str(baseline_directory),
        "-d", dataset_directory.name,
        "-c", args.configuration,
        "-f", *folds,
        "-tr", args.trainer,
        "-p", args.plans,
        "-chk", args.nnunet_checkpoint,
        "-npp", str(args.npp),
        "-nps", str(args.nps),
        "-device", "cuda" if use_cuda else "cpu",
    ]
    if args.disable_tta:
        command.append("--disable_tta")
    log("=" * 100)
    log("STAGE 1/2: method2 ORBIT-RSNA four-label segmentation")
    if use_cuda:
        log(f"Execution device: physical GPU {args.gpu} -> logical cuda:0")
    else:
        log("Execution device: CPU (CUDA is unavailable; automatic fallback)")
    log("Command: " + " ".join(command))
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"nnUNetv2_predict failed with exit code {completed.returncode}")


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def safe_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "case"


def input_case_id(path: Path) -> str:
    stem = nifti_stem(path)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return safe_case_id(stem)


def discover_batch_inputs(folder: Path) -> list[tuple[str, Path]]:
    candidates = sorted({
        path.resolve()
        for pattern in ("*.nii.gz", "*.nii")
        for path in folder.rglob(pattern)
        if path.is_file()
    })
    rows: list[tuple[str, Path]] = []
    seen: dict[str, Path] = {}
    for path in candidates:
        stem = nifti_stem(path)
        lower = stem.lower()
        if lower.endswith(("_seg", "_label", "_mask")):
            continue
        channel = re.search(r"_([0-9]{4})$", stem)
        if channel and channel.group(1) != "0000":
            continue
        case_id = input_case_id(path)
        if case_id in seen:
            raise RuntimeError(
                f"Duplicate batch case ID {case_id!r}: {seen[case_id]} and {path}. "
                "Rename one input so every output name is unique."
            )
        seen[case_id] = path
        rows.append((case_id, path))
    if not rows:
        raise RuntimeError(f"No eligible .nii or .nii.gz CT files found in {folder}")
    return rows


def stage_case(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name.lower().endswith(".nii.gz"):
        try:
            os.symlink(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return
    image = nib.load(str(source))
    nib.save(image, str(destination))


def aligned_to_reference(path: Path, reference: nib.Nifti1Image) -> nib.Nifti1Image:
    image = nib.load(str(path))
    if (
        tuple(image.shape) == tuple(reference.shape)
        and np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-4)
    ):
        return image
    return resample_from_to(
        image,
        (reference.shape, reference.affine),
        order=0,
        mode="constant",
        cval=0,
    )


def canonical_to_original(
    array: np.ndarray,
    original: nib.Nifti1Image,
    canonical: nib.Nifti1Image,
) -> np.ndarray:
    transform = ornt_transform(
        io_orientation(canonical.affine), io_orientation(original.affine)
    )
    restored = apply_orientation(array, transform)
    if tuple(restored.shape) != tuple(original.shape):
        raise RuntimeError(f"Restored shape {restored.shape} != original {original.shape}")
    return restored


def fixed_crop_3d(
    array_xyz: np.ndarray,
    center_xyz: tuple[int, int, int],
    output_dhw: tuple[int, int, int],
    fill: float,
) -> tuple[np.ndarray, tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    output_xyz = (output_dhw[2], output_dhw[1], output_dhw[0])
    output = np.full(output_xyz, fill, dtype=array_xyz.dtype)
    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for center, requested, available in zip(center_xyz, output_xyz, array_xyz.shape):
        start = int(center) - requested // 2
        stop = start + requested
        source_start = max(0, start)
        source_stop = min(available, stop)
        destination_start = source_start - start
        destination_stop = destination_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        destination_slices.append(slice(destination_start, destination_stop))
    output[tuple(destination_slices)] = array_xyz[tuple(source_slices)]
    return (
        np.transpose(output, (2, 1, 0)),
        tuple(source_slices),
        tuple(destination_slices),
    )


def resize_2d(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    factors = tuple(target / source for target, source in zip(target_hw, array.shape))
    resized = zoom(array, factors, order=1, mode="nearest", prefilter=False)
    output = np.zeros(target_hw, dtype=np.float32)
    source_slices: list[slice] = []
    destination_slices: list[slice] = []
    for available, requested in zip(resized.shape, target_hw):
        length = min(available, requested)
        source_start = (available - length) // 2
        destination_start = (requested - length) // 2
        source_slices.append(slice(source_start, source_start + length))
        destination_slices.append(slice(destination_start, destination_start + length))
    output[tuple(destination_slices)] = resized[tuple(source_slices)]
    return output


def window_to_uint8(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    clipped = np.clip(array, lower, upper)
    return np.rint((clipped - lower) * 255.0 / (upper - lower)).astype(np.uint8)


def build_global_input(
    ct: np.ndarray,
    center_z: int,
    offsets: tuple[int, ...],
    windows_hu: tuple[tuple[float, float], ...],
    global_hw: tuple[int, int],
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for offset in offsets:
        z = int(np.clip(center_z + offset, 0, ct.shape[2] - 1))
        slice_yx = ct[:, :, z].T
        resized = resize_2d(slice_yx.astype(np.float32), global_hw)
        for lower, upper in windows_hu:
            channels.append(window_to_uint8(resized, lower, upper))
    return np.stack(channels, axis=0)


def build_local_input(
    ct: np.ndarray,
    prediction: np.ndarray,
    component_mask: np.ndarray,
    center_xyz: tuple[int, int, int],
    windows_hu: tuple[tuple[float, float], ...],
    local_dhw: tuple[int, int, int],
) -> np.ndarray:
    ct_crop, _, _ = fixed_crop_3d(ct, center_xyz, local_dhw, fill=-1000.0)
    component_crop, _, _ = fixed_crop_3d(
        component_mask.astype(np.uint8), center_xyz, local_dhw, fill=0
    )
    ventricle_crop, _, _ = fixed_crop_3d(
        (prediction == VENTRICLE_LABEL).astype(np.uint8),
        center_xyz,
        local_dhw,
        fill=0,
    )
    channels = [window_to_uint8(ct_crop, *window) for window in windows_hu]
    channels.extend((component_crop * 255, ventricle_crop * 255))
    return np.stack(channels, axis=0).astype(np.uint8)


def component_features(
    ct: np.ndarray,
    component_mask: np.ndarray,
    voxel_volume_ml: float,
    center_xyz: tuple[int, int, int],
    ventricle_distance_mm: np.ndarray,
    ventricle_neighborhood: np.ndarray,
    feature_volume_normalization_ml: float,
) -> np.ndarray:
    values = ct[component_mask].astype(np.float64)
    distances = ventricle_distance_mm[component_mask]
    q10, q50, q90 = np.percentile(values, (10, 50, 90))
    adjacent = ventricle_neighborhood & component_mask
    z_fraction = center_xyz[2] / max(1, ct.shape[2] - 1)
    features = np.array(
        [
            min(
                1.0,
                component_mask.sum()
                * voxel_volume_ml
                / max(1e-6, feature_volume_normalization_ml),
            ),
            math.log1p(float(component_mask.sum())) / 12.0,
            np.clip(values.mean() / 100.0, -2.0, 3.0),
            np.clip(values.std() / 100.0, 0.0, 3.0),
            np.clip(q10 / 100.0, -2.0, 3.0),
            np.clip(q50 / 100.0, -2.0, 3.0),
            np.clip(q90 / 100.0, -2.0, 3.0),
            np.clip(float(np.min(distances)) / 50.0, 0.0, 2.0),
            np.clip(float(np.mean(distances)) / 50.0, 0.0, 2.0),
            float(adjacent.sum() / max(1, component_mask.sum())),
            float(z_fraction),
            float(np.mean(values >= 40.0)),
        ],
        dtype=np.float32,
    )
    if features.size != FEATURE_COUNT:
        raise AssertionError(features.size)
    return features


class ConvBlock3d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


class ComponentConditionedIVHRefiner(nn.Module):
    architecture_name = "CCIR_DenseNet121_2p5D_Local3D_v1"

    def __init__(self, global_channels: int, local_channels: int, feature_count: int):
        super().__init__()
        try:
            from monai.networks.nets import DenseNet121
        except ImportError as exc:
            raise RuntimeError(
                "MONAI is missing. Install dependencies from the Slicer module."
            ) from exc
        self.global_backbone = DenseNet121(
            spatial_dims=2,
            in_channels=global_channels,
            out_channels=1,
            dropout_prob=0.20,
            pretrained=False,
        )
        self.local1 = ConvBlock3d(local_channels, 32)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.local2 = ConvBlock3d(32, 64)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.local3 = ConvBlock3d(64, 128)
        self.decode2 = ConvBlock3d(128 + 64, 64)
        self.decode1 = ConvBlock3d(64 + 32, 32)
        self.residual_head = nn.Conv3d(32, 1, 1)
        with torch.no_grad():
            probe = torch.zeros(1, global_channels, 64, 64)
            global_features = int(self.global_backbone.features(probe).shape[1])
        self.classifier = nn.Sequential(
            nn.Linear(global_features + 128 + feature_count, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        global_input: torch.Tensor,
        local_input: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_feature = torch.relu(self.global_backbone.features(global_input))
        global_feature = torch.nn.functional.adaptive_avg_pool2d(
            global_feature, 1
        ).flatten(1)
        local1 = self.local1(local_input)
        local2 = self.local2(self.pool1(local1))
        local3 = self.local3(self.pool2(local2))
        local_feature = torch.nn.functional.adaptive_avg_pool3d(local3, 1).flatten(1)
        up2 = torch.nn.functional.interpolate(
            local3, size=local2.shape[2:], mode="trilinear", align_corners=False
        )
        decode2 = self.decode2(torch.cat((up2, local2), dim=1))
        up1 = torch.nn.functional.interpolate(
            decode2, size=local1.shape[2:], mode="trilinear", align_corners=False
        )
        decode1 = self.decode1(torch.cat((up1, local1), dim=1))
        residual_logits = self.residual_head(decode1)
        component_logits = self.classifier(
            torch.cat((global_feature, local_feature, features), dim=1)
        ).squeeze(1)
        return component_logits, residual_logits


def clean_key(key: str) -> str:
    previous = None
    while previous != key:
        previous = key
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
    return key


def checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for field in ("model_state_dict", "state_dict", "network_weights"):
        value = checkpoint.get(field)
        if isinstance(value, dict):
            return {
                clean_key(str(key)): tensor
                for key, tensor in value.items()
                if torch.is_tensor(tensor)
            }
    raise RuntimeError("CCIR checkpoint does not contain a model state dictionary")


def validate_ccir_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    """Reject RSNA classifier files and incompatible/old CCIR architectures."""
    architecture = checkpoint.get("architecture")
    if architecture is not None and architecture != EXPECTED_CCIR_ARCHITECTURE:
        raise RuntimeError(
            f"Wrong CCIR architecture in {path}: {architecture!r}; expected "
            f"{EXPECTED_CCIR_ARCHITECTURE!r}."
        )
    state = checkpoint_state(checkpoint)
    keys = tuple(state)
    required_prefixes = (
        "global_backbone.",
        "local1.",
        "local2.",
        "local3.",
        "decode2.",
        "decode1.",
        "residual_head.",
        "classifier.",
    )
    missing = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        classifier_like = any(
            key.startswith(("network.features.", "network.class_layers."))
            for key in keys
        )
        hint = (
            " This appears to be RSNA classifier_best.pt. Select "
            "orbit_rsna_ccir_all_train_all_validate/ccir_best_resubstitution.pt instead."
            if classifier_like else ""
        )
        raise RuntimeError(
            "Selected file is not a complete CCIR deployment checkpoint; missing "
            f"state prefixes: {missing}.{hint}"
        )
    baseline = checkpoint.get("baseline")
    if isinstance(baseline, dict):
        method = str(baseline.get("method", ""))
        if method and "ORBIT" not in method.upper():
            raise RuntimeError(
                "The CCIR checkpoint records a non-ORBIT baseline and must not be "
                "combined with the method2 ORBIT-RSNA segmentation model."
            )
    log(f"CCIR checkpoint preflight passed: {path}")


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tuple_int(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def tuple_windows(
    value: Any,
    default: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if value is None:
        return default
    return tuple((float(item[0]), float(item[1])) for item in value)


def resolve_probability_thresholds(
    args: argparse.Namespace,
) -> tuple[float, float]:
    remove = float(args.remove_threshold)
    keep = float(args.keep_refine_threshold)
    if not 0 <= remove <= 1 or not 0 <= keep <= 1 or keep < remove:
        raise ValueError(
            f"Invalid probability thresholds: remove={remove}, keep/refine={keep}"
        )
    return remove, keep


def place_local_prediction(
    final: np.ndarray,
    component_mask: np.ndarray,
    residual_probability_dhw: np.ndarray,
    center_xyz: tuple[int, int, int],
    local_dhw: tuple[int, int, int],
    residual_voxel_threshold: float,
    dilation_iterations: int,
) -> int:
    component_crop, source, destination = fixed_crop_3d(
        component_mask.astype(np.uint8), center_xyz, local_dhw, fill=0
    )
    support = binary_dilation(
        component_crop.astype(bool), iterations=dilation_iterations
    )
    proposed_dhw = (residual_probability_dhw >= residual_voxel_threshold) & support
    proposed_xyz = np.transpose(proposed_dhw, (2, 1, 0))
    proposed_valid = proposed_xyz[destination]
    current = final[source]
    add = proposed_valid & (current == 0)
    current[add] = IVH_LABEL
    final[source] = current
    return int(add.sum())


def initialize_ccir_runtime(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Load the CCIR network once and reuse it for every batch case."""
    settings = checkpoint.get("settings", {})
    global_offsets = tuple_int(settings.get("global_offsets"), DEFAULT_GLOBAL_OFFSETS)
    windows_hu = tuple_windows(settings.get("windows_hu"), DEFAULT_WINDOWS_HU)
    global_hw = tuple_int(settings.get("global_hw"), DEFAULT_GLOBAL_HW)
    local_dhw = tuple_int(settings.get("local_dhw"), DEFAULT_LOCAL_DHW)
    feature_volume_normalization_ml = float(
        settings.get("candidate_max_volume_ml", args.component_volume_threshold_ml)
    )
    residual_voxel_threshold = float(
        settings.get("residual_voxel_threshold", DEFAULT_RESIDUAL_VOXEL_THRESHOLD)
    )
    dilation_iterations = int(
        settings.get(
            "residual_support_dilation_iterations",
            DEFAULT_RESIDUAL_DILATION_ITERATIONS,
        )
    )
    remove_threshold, keep_threshold = resolve_probability_thresholds(args)
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    model = ComponentConditionedIVHRefiner(
        global_channels=len(global_offsets) * len(windows_hu),
        local_channels=len(windows_hu) + 2,
        feature_count=FEATURE_COUNT,
    )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.to(device).eval()
    log("=" * 100)
    log("STAGE 2/2: Component-conditioned IVH refinement")
    log(f"Device: {device} ({device_name})")
    log(f"Component volume gate: < {args.component_volume_threshold_ml:.4f} mL")
    log(
        "Feature volume normalization: "
        f"{feature_volume_normalization_ml:.4f} mL (from training checkpoint)"
    )
    log(f"GUI remove threshold: {remove_threshold:.6f}")
    log(f"GUI keep/refine threshold: {keep_threshold:.6f}")
    return {
        "model": model,
        "device": device,
        "device_name": device_name,
        "global_offsets": global_offsets,
        "windows_hu": windows_hu,
        "global_hw": global_hw,
        "local_dhw": local_dhw,
        "feature_volume_normalization_ml": feature_volume_normalization_ml,
        "residual_voxel_threshold": residual_voxel_threshold,
        "dilation_iterations": dilation_iterations,
        "remove_threshold": remove_threshold,
        "keep_threshold": keep_threshold,
    }


def refine_ivh(
    args: argparse.Namespace,
    input_path: Path,
    baseline_path: Path,
    output_path: Path,
    case_id: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    original = nib.load(str(input_path))
    canonical_image = nib.as_closest_canonical(original)
    ct = np.asarray(canonical_image.dataobj, dtype=np.float32)
    prediction_image = nib.as_closest_canonical(
        aligned_to_reference(baseline_path, original)
    )
    prediction = np.rint(np.asarray(prediction_image.dataobj)).astype(np.uint8)
    if ct.ndim != 3 or prediction.shape != ct.shape:
        raise ValueError(f"Incompatible CT/prediction shapes: {ct.shape}, {prediction.shape}")
    if not np.isfinite(ct).all():
        raise ValueError("Input CT contains NaN or Inf")
    unexpected = np.setdiff1d(np.unique(prediction), np.arange(5))
    if unexpected.size:
        raise ValueError(f"nnU-Net produced unexpected labels: {unexpected.tolist()}")

    model = runtime["model"]
    device = runtime["device"]
    device_name = runtime["device_name"]
    global_offsets = runtime["global_offsets"]
    windows_hu = runtime["windows_hu"]
    global_hw = runtime["global_hw"]
    local_dhw = runtime["local_dhw"]
    feature_volume_normalization_ml = runtime["feature_volume_normalization_ml"]
    residual_voxel_threshold = runtime["residual_voxel_threshold"]
    dilation_iterations = runtime["dilation_iterations"]
    remove_threshold = runtime["remove_threshold"]
    keep_threshold = runtime["keep_threshold"]
    log(f"Refining case: {case_id}")

    zooms_mm = tuple(float(value) for value in canonical_image.header.get_zooms()[:3])
    voxel_volume_ml = float(np.prod(zooms_mm) / 1000.0)
    ventricle_mask = prediction == VENTRICLE_LABEL
    if np.any(ventricle_mask):
        ventricle_distance_mm = distance_transform_edt(
            ~ventricle_mask, sampling=zooms_mm
        ).astype(np.float32)
        ventricle_neighborhood = binary_dilation(ventricle_mask, iterations=2)
    else:
        ventricle_distance_mm = np.full(prediction.shape, 100.0, dtype=np.float32)
        ventricle_neighborhood = np.zeros(prediction.shape, dtype=bool)

    structure = generate_binary_structure(3, 3)
    components, component_count = connected_components(
        prediction == IVH_LABEL, structure=structure
    )
    final = prediction.copy()
    actions: list[dict[str, Any]] = []
    checked = removed = refined = 0
    removed_voxels = added_voxels = 0

    for component_id in range(1, component_count + 1):
        component_mask = components == component_id
        voxel_count = int(component_mask.sum())
        volume_ml = float(voxel_count * voxel_volume_ml)
        action: dict[str, Any] = {
            "component_id": component_id,
            "component_voxels": voxel_count,
            "component_volume_ml": volume_ml,
        }
        if voxel_count == 0:
            continue
        if volume_ml >= args.component_volume_threshold_ml:
            action.update(
                probability_true_component=None,
                action="preserve_at_or_above_volume_gate",
                removed_voxels=0,
                added_voxels=0,
            )
            actions.append(action)
            continue

        checked += 1
        center_float = center_of_mass(component_mask)
        center_xyz = tuple(int(round(value)) for value in center_float)
        global_uint8 = build_global_input(
            ct, center_xyz[2], global_offsets, windows_hu, global_hw
        )
        local_uint8 = build_local_input(
            ct, prediction, component_mask, center_xyz, windows_hu, local_dhw
        )
        features_np = component_features(
            ct,
            component_mask,
            voxel_volume_ml,
            center_xyz,
            ventricle_distance_mm,
            ventricle_neighborhood,
            feature_volume_normalization_ml,
        )
        global_input = torch.from_numpy(
            global_uint8.astype(np.float32) / 127.5 - 1.0
        )[None].to(device)
        local_np = local_uint8.astype(np.float32)
        local_np[: len(windows_hu)] = local_np[: len(windows_hu)] / 127.5 - 1.0
        local_np[len(windows_hu) :] = local_np[len(windows_hu) :] / 255.0
        local_input = torch.from_numpy(local_np)[None].to(device)
        feature_input = torch.from_numpy(features_np)[None].to(device)
        autocast_context = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            component_logits, residual_logits = model(
                global_input, local_input, feature_input
            )
            probability = float(torch.sigmoid(component_logits)[0].float().cpu())
            residual_probability = (
                torch.sigmoid(residual_logits)[0, 0].float().cpu().numpy()
            )

        component_removed = component_added = 0
        if probability < remove_threshold:
            final[component_mask] = 0
            component_removed = voxel_count
            removed += 1
            removed_voxels += component_removed
            decision = "remove_to_background"
        elif (
            not args.disable_residual_addition
            and probability >= keep_threshold
        ):
            component_added = place_local_prediction(
                final,
                component_mask,
                residual_probability,
                center_xyz,
                local_dhw,
                residual_voxel_threshold,
                dilation_iterations,
            )
            refined += 1
            added_voxels += component_added
            decision = "keep_and_add_residual"
        else:
            decision = "abstain_keep_method2"
        action.update(
            center_xyz=list(center_xyz),
            probability_true_component=probability,
            action=decision,
            removed_voxels=component_removed,
            added_voxels=component_added,
        )
        actions.append(action)
        log(
            f"Component {component_id}/{component_count}: {volume_ml:.4f} mL, "
            f"P(true IVH)={probability:.5f}, action={decision}"
        )

    # Hard safety invariant: the CCIR stage may change only background/IVH.
    for protected_label in (1, 2, 3):
        if not np.array_equal(
            final == protected_label, prediction == protected_label
        ):
            raise RuntimeError(
                f"CCIR changed protected label {protected_label}; output was not saved"
            )

    restored = canonical_to_original(final, original, canonical_image).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(restored, original.affine, header), str(output_path))

    report = {
        "method": "method2 ORBIT-RSNA four-label segmentation + method1 CCIR",
        "case_id": case_id,
        "input": str(input_path),
        "baseline_prediction": str(baseline_path),
        "output": str(output_path),
        "labels": {str(key): value for key, value in LABEL_NAMES.items()},
        "device": str(device),
        "device_name": device_name,
        "component_connectivity": 26,
        "component_volume_gate_ml": float(args.component_volume_threshold_ml),
        "component_gate_rule": "check only if component_volume_ml < gate",
        "feature_volume_normalization_ml": feature_volume_normalization_ml,
        "remove_probability_threshold": remove_threshold,
        "keep_refine_probability_threshold": keep_threshold,
        "calibration_source": "GUI/command-line settings",
        "residual_addition_enabled": not args.disable_residual_addition,
        "predicted_ivh_components_total": int(component_count),
        "components_checked_by_ccir": checked,
        "components_removed": removed,
        "components_refined": refined,
        "removed_ivh_voxels": removed_voxels,
        "added_ivh_voxels": added_voxels,
        "voxel_volume_ml": voxel_volume_ml,
        "actions": actions,
        "important": (
            "Labels 1 (ICH), 2 (Edema), and 3 (Ventricle) are copied from "
            "method2 ORBIT-RSNA. "
            "CCIR modifies only label 4 and background."
        ),
    }
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )


def run_single_case(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    started = time.time()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CT not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Add --overwrite to replace it."
        )
    case_id = safe_case_id(args.case_id) if args.case_id else input_case_id(input_path)
    with tempfile.TemporaryDirectory(prefix="slicer_orbit_ccir_single_") as temp:
        baseline_directory = Path(temp) / "orbit_prediction"
        run_nnunet(args, input_path.parent, baseline_directory)
        input_stem = nifti_stem(input_path)
        if not input_stem.endswith("_0000"):
            raise ValueError(f"Single input must end in _0000: {input_path.name}")
        baseline_path = baseline_directory / f"{input_stem[:-5]}.nii.gz"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"ORBIT prediction not found: {baseline_path}")
        runtime = initialize_ccir_runtime(args, checkpoint)
        report = refine_ivh(
            args, input_path, baseline_path, output_path, case_id, runtime
        )
    report["elapsed_seconds"] = time.time() - started
    write_json(report_path, report)
    log("=" * 100)
    log(f"Final segmentation: {output_path}")
    log(f"Inference report: {report_path}")
    log(f"Elapsed: {report['elapsed_seconds']:.1f} seconds")


def run_batch(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    started = time.time()
    input_folder = args.input_folder.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    cases = discover_batch_inputs(input_folder)
    pending: list[tuple[str, Path]] = []
    results: list[dict[str, Any]] = []
    for case_id, input_path in cases:
        output_path = output_folder / f"{case_id}_seg.nii.gz"
        report_path = output_folder / f"{case_id}_ccir_report.json"
        if output_path.is_file() and not args.overwrite:
            results.append({
                "case_id": case_id,
                "input": str(input_path),
                "output": str(output_path),
                "report": str(report_path),
                "status": "skipped_existing",
            })
        else:
            pending.append((case_id, input_path))

    log("=" * 100)
    log("BATCH INPUT")
    log(f"Discovered: {len(cases)}")
    log(f"Pending: {len(pending)}")
    log(f"Skipped existing: {len(cases) - len(pending)}")

    if pending:
        with tempfile.TemporaryDirectory(prefix="slicer_orbit_ccir_batch_") as temp:
            work = Path(temp)
            staged_input = work / "input"
            baseline_directory = work / "orbit_prediction"
            for case_id, input_path in pending:
                stage_case(input_path, staged_input / f"{case_id}_0000.nii.gz")
            run_nnunet(args, staged_input, baseline_directory)
            runtime = initialize_ccir_runtime(args, checkpoint)
            for number, (case_id, input_path) in enumerate(pending, 1):
                case_started = time.time()
                output_path = output_folder / f"{case_id}_seg.nii.gz"
                report_path = output_folder / f"{case_id}_ccir_report.json"
                baseline_path = baseline_directory / f"{case_id}.nii.gz"
                log("=" * 100)
                log(f"BATCH CASE {number}/{len(pending)}: {case_id}")
                try:
                    if not baseline_path.is_file():
                        raise FileNotFoundError(
                            f"ORBIT prediction not found: {baseline_path}"
                        )
                    report = refine_ivh(
                        args,
                        input_path,
                        baseline_path,
                        output_path,
                        case_id,
                        runtime,
                    )
                    report["elapsed_seconds"] = time.time() - case_started
                    write_json(report_path, report)
                    results.append({
                        "case_id": case_id,
                        "input": str(input_path),
                        "output": str(output_path),
                        "report": str(report_path),
                        "status": "completed",
                        "elapsed_seconds": report["elapsed_seconds"],
                    })
                except Exception as error:
                    import traceback

                    traceback.print_exc()
                    results.append({
                        "case_id": case_id,
                        "input": str(input_path),
                        "output": str(output_path),
                        "report": str(report_path),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    })

    completed = sum(item["status"] == "completed" for item in results)
    skipped = sum(item["status"] == "skipped_existing" for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    summary = {
        "method": "method2 ORBIT-RSNA + method1 CCIR",
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "discovered": len(cases),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "component_volume_gate_ml": args.component_volume_threshold_ml,
        "remove_threshold_gui": args.remove_threshold,
        "keep_refine_threshold_gui": args.keep_refine_threshold,
        "preferred_physical_gpu": args.gpu,
        "actual_device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "elapsed_seconds": time.time() - started,
        "cases": sorted(results, key=lambda item: item["case_id"]),
    }
    summary_path = output_folder / "orbit_ccir_batch_summary.json"
    write_json(summary_path, summary)
    log("=" * 100)
    log(f"Batch summary: {summary_path}")
    log(f"Completed={completed}, skipped={skipped}, failed={failed}")
    if failed:
        raise RuntimeError(
            f"Batch completed with {failed} failed case(s). See {summary_path}"
        )


def main() -> None:
    args = parse_args()
    args.nnunet_model_dir = args.nnunet_model_dir.expanduser().resolve()
    args.ccir_checkpoint = args.ccir_checkpoint.expanduser().resolve()
    if not args.ccir_checkpoint.is_file():
        raise FileNotFoundError(f"CCIR checkpoint not found: {args.ccir_checkpoint}")
    # Check every dependency and the CCIR file before expensive segmentation.
    verify_runtime_dependencies()
    install_bundled_orbit_trainer(args.trainer)
    ccir_checkpoint = load_torch_checkpoint(args.ccir_checkpoint)
    validate_ccir_checkpoint(ccir_checkpoint, args.ccir_checkpoint)
    if args.input is not None:
        run_single_case(args, ccir_checkpoint)
    else:
        run_batch(args, ccir_checkpoint)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback

        log(f"FATAL: {type(error).__name__}: {error}")
        traceback.print_exc()
        raise SystemExit(1)
