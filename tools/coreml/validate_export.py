from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import coremltools as ct
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from export_camera_token import (
    DA3ImageInputWrapper,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_SOURCE,
    load_api_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a DA3 Small Core ML export against untouched PyTorch."
    )
    parser.add_argument("--model", required=True, help="Core ML package or compiled model.")
    parser.add_argument("--image", required=True, help="Reference image path.")
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--compute-unit",
        choices=("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"),
        default="CPU_AND_GPU",
    )
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--timed-runs", type=int, default=20)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-mean-abs-diff", type=float, default=0.01)
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
    }


def main() -> int:
    args = parse_args()
    if platform.system() != "Darwin":
        raise RuntimeError("Core ML prediction validation requires macOS.")

    model_path = Path(args.model).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    image = Image.open(image_path).convert("RGB").resize(
        (args.input_size, args.input_size),
        Image.Resampling.BICUBIC,
    )
    image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
    input_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    official_model = load_api_model(
        "da3-small",
        args.model_source,
        args.model_revision,
        args.model_sha256,
    )
    official_wrapper = DA3ImageInputWrapper(official_model).eval()
    with torch.inference_mode():
        reference = official_wrapper(input_tensor).detach().cpu().float().squeeze()
    del official_wrapper
    del official_model

    compute_unit = getattr(ct.ComputeUnit, args.compute_unit)
    coreml_model = ct.models.MLModel(str(model_path), compute_units=compute_unit)
    specification = coreml_model.get_spec()
    input_name = specification.description.input[0].name
    output_name = specification.description.output[0].name

    for _ in range(args.warmup_runs):
        coreml_model.predict({input_name: image})

    timings = []
    prediction = None
    for _ in range(args.timed_runs):
        start = time.perf_counter()
        prediction = coreml_model.predict({input_name: image})[output_name]
        timings.append((time.perf_counter() - start) * 1000.0)

    if prediction is None:
        raise RuntimeError("At least one timed prediction is required.")

    candidate = torch.from_numpy(np.asarray(prediction, dtype=np.float32).copy()).squeeze()
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    if reference_flat.shape != candidate_flat.shape:
        raise RuntimeError(
            f"Output shape mismatch: PyTorch {tuple(reference.shape)}, "
            f"Core ML {tuple(candidate.shape)}"
        )

    difference = (reference_flat - candidate_flat).abs()
    cosine = float(F.cosine_similarity(reference_flat, candidate_flat, dim=0).item())
    cosine = max(-1.0, min(1.0, cosine))
    mean_abs_diff = float(difference.mean().item())
    result = {
        "model": str(model_path),
        "image": str(image_path),
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "model_sha256": args.model_sha256,
        "compute_unit": args.compute_unit,
        "pytorch": tensor_summary(reference),
        "coreml": tensor_summary(candidate),
        "cosine_similarity": cosine,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": float(difference.max().item()),
        "median_prediction_ms": statistics.median(timings),
        "timed_runs": args.timed_runs,
    }
    print(json.dumps(result, indent=2))

    if cosine < args.minimum_cosine:
        print(
            f"Cosine similarity {cosine:.8f} is below {args.minimum_cosine:.8f}.",
            flush=True,
        )
        return 1
    if mean_abs_diff > args.maximum_mean_abs_diff:
        print(
            f"Mean absolute difference {mean_abs_diff:.8f} exceeds "
            f"{args.maximum_mean_abs_diff:.8f}.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
