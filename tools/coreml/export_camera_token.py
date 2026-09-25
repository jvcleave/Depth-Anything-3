from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY


DEFAULT_MODEL_NAME = "da3-small"
DEFAULT_MODEL_SOURCE = "depth-anything/DA3-SMALL"
DEFAULT_MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
DEFAULT_MODEL_SHA256 = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
DEFAULT_OUTPUT = "build/coreml/DepthAnything3SmallCameraToken.mlpackage"
SAFETENSORS_NAME = "model.safetensors"

class TraceablePositionGetter:
    """Core ML-compatible equivalent of torch.cartesian_prod's row-major grid."""

    def __call__(self, batch_size, height, width, device):
        y = torch.arange(height, device=device)
        x = torch.arange(width, device=device)
        grid_y = y.unsqueeze(1).expand(height, width).reshape(-1)
        grid_x = x.unsqueeze(0).expand(height, width).reshape(-1)
        positions = torch.stack([grid_y, grid_x], dim=-1)
        return positions.unsqueeze(0).expand(batch_size, -1, -1)


def patch_camera_token_assignment() -> None:
    """Replace DA3's in-place camera-token write with an equivalent concat."""
    import depth_anything_3.model.dinov2.vision_transformer as vt

    def patched_get_intermediate_layers_not_chunked(
        self, x, n=1, export_feat_layers=[], **kwargs
    ):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output = []
        aux_output = []
        total_block_len = len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos, l_pos = pos_nodiff, pos

            if self.alt_start != -1 and i == self.alt_start:
                supplied_camera_token = kwargs.get("cam_token", None)
                if supplied_camera_token is not None:
                    camera_token = supplied_camera_token
                else:
                    # This exporter has a fixed S=1 contract. Avoid expressing
                    # the official zero-length source-token branch because
                    # coremltools incorrectly infers it as one element.
                    camera_token = self.camera_token[:, :1].expand(B, S, -1)
                x = torch.cat([camera_token.unsqueeze(2), x[:, :, 1:, :]], dim=2)

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)

        return output, aux_output

    vt.DinoVisionTransformer._get_intermediate_layers_not_chunked = (
        patched_get_intermediate_layers_not_chunked
    )


def patch_uv_grid() -> None:
    """Replace meshgrid with an exactly ordered Core ML-compatible UV grid."""
    from depth_anything_3.model.utils import head_utils
    import depth_anything_3.model.dualdpt as dualdpt_module

    def patched_create_uv_grid(width, height, aspect_ratio=None, dtype=None, device=None):
        if aspect_ratio is None:
            aspect_ratio = float(width) / float(height)
        diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
        span_x = aspect_ratio / diag_factor
        span_y = 1.0 / diag_factor
        left_x = -span_x * (width - 1) / width
        right_x = span_x * (width - 1) / width
        top_y = -span_y * (height - 1) / height
        bottom_y = span_y * (height - 1) / height
        x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
        y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
        # torch.meshgrid(x_coords, y_coords, indexing="xy") produces H x W.
        uu = x_coords.reshape(1, width).repeat(height, 1)
        vv = y_coords.reshape(height, 1).repeat(1, width)
        return torch.stack((uu, vv), dim=-1)

    head_utils.create_uv_grid = patched_create_uv_grid
    dualdpt_module.create_uv_grid = patched_create_uv_grid


class LocalDepthAnything3(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()


class DA3DepthOnlyWrapper(torch.nn.Module):
    def __init__(
        self,
        api_model: LocalDepthAnything3,
        ref_view_strategy: str = "saddle_balanced",
    ) -> None:
        super().__init__()
        self.api_model = api_model
        self.net = api_model.model
        self.ref_view_strategy = ref_view_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        h, w = x.shape[-2:]
        x5 = x.unsqueeze(1)
        feats, _ = self.net.backbone(
            x5,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy,
        )
        output = self.net.head(feats, h, w, patch_start_idx=0)
        depth = output["depth"]
        # (B, 1, H, W) — keep channel dim for ImageType grayscale output compat
        return depth[:, 0:1]


class DA3ImageInputWrapper(torch.nn.Module):
    """Wrapper that bakes exact ImageNet normalization into the model.

    Accepts 0-1 range input (Core ML ImageType with scale=1/255 provides this).
    Applies exact per-channel (x - mean) / std before the backbone.
    This avoids the single-scalar-scale limitation of Core ML ImageType
    and keeps numerical results identical to the tensor-input path.
    """

    def __init__(
        self,
        api_model: LocalDepthAnything3,
        ref_view_strategy: str = "saddle_balanced",
    ) -> None:
        super().__init__()
        self.net = api_model.model
        self.ref_view_strategy = ref_view_strategy
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 3, H, W) in 0-1 range from Core ML ImageType
        x = x.float()
        x = (x - self.mean) / self.std
        h, w = x.shape[-2:]
        x5 = x.unsqueeze(1)
        feats, _ = self.net.backbone(
            x5,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy,
        )
        output = self.net.head(feats, h, w, patch_start_idx=0)
        depth = output["depth"]
        # (B, 1, H, W) — keep channel dim for ImageType grayscale output compat
        return depth[:, 0:1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Depth Anything 3 to Core ML.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-source",
        default=DEFAULT_MODEL_SOURCE,
        help="Hugging Face repository or local model path.",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Pinned Hugging Face commit used when model-source is a repository.",
    )
    parser.add_argument(
        "--model-sha256",
        default=DEFAULT_MODEL_SHA256,
        help="Expected model.safetensors SHA-256; pass an empty value to skip the check.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output .mlpackage path.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Square input size for tracing/conversion.",
    )
    parser.add_argument(
        "--use-image-input",
        action="store_true",
        help="Use Core ML ImageType input instead of TensorType.",
    )
    parser.add_argument(
        "--trace-output",
        default="build/coreml/DepthAnything3SmallCameraToken_traced.pt",
        help="Optional TorchScript output path.",
    )
    parser.add_argument(
        "--grayscale-output",
        action="store_true",
        help="Output as Grayscale16Half ImageType (matches V2 format) instead of MultiArray.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=("float16", "float32"),
        default="float16",
        help="Core ML internal compute precision (default: float16).",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Stop after eager + trace validation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def resolve_weights_path(
    model_source: str | None,
    model_revision: str | None,
    expected_sha256: str | None,
) -> str | None:
    if not model_source:
        return None
    source_path = Path(model_source)
    if source_path.exists():
        if source_path.is_dir():
            candidate = source_path / SAFETENSORS_NAME
            if not candidate.exists():
                raise FileNotFoundError(f"Expected {candidate} to exist.")
            weights_path = candidate
        else:
            weights_path = source_path
    else:
        weights_path = Path(
            hf_hub_download(
                repo_id=model_source,
                filename=SAFETENSORS_NAME,
                revision=model_revision,
            )
        )

    if expected_sha256:
        digest = sha256_file(weights_path)
        if digest != expected_sha256:
            raise RuntimeError(
                f"Unexpected weights SHA-256 for {weights_path}: "
                f"expected {expected_sha256}, got {digest}"
            )
    return str(weights_path)


def load_api_model(
    model_name: str,
    model_source: str | None,
    model_revision: str | None,
    model_sha256: str | None,
) -> LocalDepthAnything3:
    model = LocalDepthAnything3(model_name=model_name)
    weights_path = resolve_weights_path(model_source, model_revision, model_sha256)
    if weights_path is not None:
        state_dict = load_file(weights_path, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "model.head.scratch.output_conv2_aux.",
        )
        disallowed_missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "State dict mismatch. "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        if missing:
            print(f"Allowing missing auxiliary-head weights: {missing}")
    model.eval()
    model.to("cpu")
    model.float()
    return model


def build_example_input(input_size: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, input_size, input_size, dtype=torch.float32)


def trace_wrapper(wrapper: torch.nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
    with torch.inference_mode():
        return torch.jit.trace(wrapper, example_input, strict=False)


def make_coreml_input(input_size: int, use_image_input: bool) -> list[Any]:
    if use_image_input:
        # scale only: 0-255 -> 0-1.  Per-channel ImageNet normalization is
        # baked into DA3ImageInputWrapper so there is no approximation error.
        return [
            ct.ImageType(
                name="image",
                shape=(1, 3, input_size, input_size),
                scale=1 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ]
    return [ct.TensorType(name="image", shape=(1, 3, input_size, input_size))]


def make_coreml_output(input_size: int, grayscale_output: bool) -> list[Any]:
    if grayscale_output:
        return [
            ct.ImageType(
                name="depth",
                color_layout=ct.colorlayout.GRAYSCALE_FLOAT16,
            )
        ]
    return [ct.TensorType(name="depth")]


def convert_traced_model(
    traced: torch.jit.ScriptModule,
    input_size: int,
    use_image_input: bool,
    grayscale_output: bool,
    compute_precision: str,
    output_path: Path,
) -> ct.models.MLModel:
    precision = (
        ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
    )
    mlmodel = ct.convert(
        traced,
        inputs=make_coreml_input(input_size, use_image_input),
        outputs=make_coreml_output(input_size, grayscale_output),
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
        compute_precision=precision,
    )
    mlmodel.save(str(output_path))
    return mlmodel


def summarize_tensor(t: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_flat = reference.float().reshape(-1)
    candidate_flat = candidate.float().reshape(-1)
    cosine_similarity = float(
        F.cosine_similarity(reference_flat, candidate_flat, dim=0).item()
    )
    return {
        "max_abs_diff": float((reference_flat - candidate_flat).abs().max().item()),
        "mean_abs_diff": float((reference_flat - candidate_flat).abs().mean().item()),
        "cosine_similarity": max(-1.0, min(1.0, cosine_similarity)),
    }


# Models that use RoPE and need monkey-patching for coremltools
ROPE_MODELS = {"da3-small", "da3-base", "da3-large", "da3-giant"}


def needs_monkey_patches(model_name: str) -> bool:
    return model_name in ROPE_MODELS


def apply_rope_model_fixups(api_model: LocalDepthAnything3) -> None:
    """Install the traceable position grid while preserving alt_start."""
    vit = api_model.model.backbone.pretrained
    if hasattr(vit, "position_getter") and vit.position_getter is not None:
        vit.position_getter = TraceablePositionGetter()


def freeze_position_embedding(api_model: LocalDepthAnything3, input_size: int) -> None:
    """Precompute DINOv2 positional interpolation for the fixed export shape."""
    vit = api_model.model.backbone.pretrained
    patch_size = int(vit.patch_size)
    patch_count = (input_size // patch_size) ** 2
    native_patch_count = vit.pos_embed.shape[1] - 1
    if patch_count == native_patch_count:
        return

    placeholder = torch.empty(
        1,
        patch_count + 1,
        vit.embed_dim,
        dtype=vit.pos_embed.dtype,
        device=vit.pos_embed.device,
    )
    with torch.inference_mode():
        fixed_pos_embed = vit.interpolate_pos_encoding(
            placeholder,
            input_size,
            input_size,
        ).detach()
    vit.pos_embed = nn.Parameter(fixed_pos_embed, requires_grad=False)


def main() -> int:
    args = parse_args()
    if args.input_size <= 0 or args.input_size % 14 != 0:
        raise ValueError(
            "--input-size must be a positive multiple of DA3's 14-pixel patch size"
        )
    output_path = (REPO_ROOT / args.output).resolve()
    trace_path = (REPO_ROOT / args.trace_output).resolve()
    example_input = build_example_input(args.input_size)

    official_output = None
    camera_functional_output = None
    if needs_monkey_patches(args.model_name):
        print(f"Running untouched {args.model_name} PyTorch oracle...")
        official_model = load_api_model(
            args.model_name,
            args.model_source,
            args.model_revision,
            args.model_sha256,
        )
        if args.use_image_input:
            official_wrapper = DA3ImageInputWrapper(official_model)
        else:
            official_wrapper = DA3DepthOnlyWrapper(official_model)
        official_wrapper.eval()
        with torch.inference_mode():
            official_output = official_wrapper(example_input).detach().clone()
        del official_wrapper
        del official_model
        gc.collect()

        print("Checking functional camera-token replacement in isolation...")
        patch_camera_token_assignment()
        camera_functional_model = load_api_model(
            args.model_name,
            args.model_source,
            args.model_revision,
            args.model_sha256,
        )
        freeze_position_embedding(camera_functional_model, args.input_size)
        if args.use_image_input:
            camera_functional_wrapper = DA3ImageInputWrapper(camera_functional_model)
        else:
            camera_functional_wrapper = DA3DepthOnlyWrapper(camera_functional_model)
        camera_functional_wrapper.eval()
        with torch.inference_mode():
            camera_functional_output = camera_functional_wrapper(example_input).detach().clone()
        del camera_functional_wrapper
        del camera_functional_model
        gc.collect()

    if needs_monkey_patches(args.model_name):
        print("Applying the exact Core ML-compatible UV-grid replacement...")
        patch_uv_grid()

    api_model = load_api_model(
        args.model_name,
        args.model_source,
        args.model_revision,
        args.model_sha256,
    )

    if needs_monkey_patches(args.model_name):
        freeze_position_embedding(api_model, args.input_size)
        apply_rope_model_fixups(api_model)

    if args.use_image_input:
        wrapper = DA3ImageInputWrapper(api_model)
    else:
        wrapper = DA3DepthOnlyWrapper(api_model)
    wrapper.eval()

    with torch.inference_mode():
        eager_output = wrapper(example_input)

    traced = trace_wrapper(wrapper, example_input)
    with torch.inference_mode():
        traced_output = traced(example_input)

    max_abs_diff = float((eager_output - traced_output).abs().max().item())
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(trace_path))

    summary = {
        "model_name": args.model_name,
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "model_sha256": args.model_sha256,
        "input_size": args.input_size,
        "use_image_input": args.use_image_input,
        "compute_precision": args.compute_precision,
        "eager": summarize_tensor(eager_output),
        "traced": summarize_tensor(traced_output),
        "trace_max_abs_diff": max_abs_diff,
        "trace_path": str(trace_path),
    }
    if official_output is not None:
        summary["official_vs_camera_functional"] = compare_tensors(
            official_output, camera_functional_output
        )
        summary["official_vs_functional"] = compare_tensors(official_output, eager_output)
        summary["official_vs_traced"] = compare_tensors(official_output, traced_output)

    if args.skip_conversion:
        print(json.dumps(summary, indent=2))
        return 0

    mlmodel = convert_traced_model(
        traced=traced,
        input_size=args.input_size,
        use_image_input=args.use_image_input,
        grayscale_output=args.grayscale_output,
        compute_precision=args.compute_precision,
        output_path=output_path,
    )
    summary["coreml_output"] = str(output_path)
    summary["coreml_short_description"] = mlmodel.short_description or ""
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
