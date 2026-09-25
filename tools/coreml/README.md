# DA3 Small Core ML Camera-Token Export

## Why this exporter exists

The first DA3 Small Core ML experiment set the backbone's `alt_start` to `-1`
because the original camera-token insertion uses an in-place tensor assignment.
That removed both the learned camera token and the alternating global-attention
blocks that follow it. The resulting model converted and ran quickly, but its
output did not reliably behave like coherent scene depth.

`tools/coreml/export_camera_token.py` preserves the official DA3 Small
single-view path. At `S = 1`, the model uses its learned reference camera token;
it does not need camera calibration or a live camera input. Global cross-view
attention naturally reduces to self-attention for the one input view.

This is an unofficial conversion maintained by the MESS project. It is based
on ByteDance Seed `main` commit `3d835ec` and keeps its source code and DA3
Small model under their Apache 2.0 licenses.

## Core ML-compatible rewrites

The exporter makes three fixed-shape rewrites:

1. Replace `x[:, :, 0] = camera_token` with concatenation of the learned token
   and `x[:, :, 1:, :]`.
2. Replace `torch.cartesian_prod` with an explicitly expanded row-major position
   grid.
3. Replace the DualDPT `meshgrid` with repeated coordinate vectors while
   preserving the original `indexing="xy"` axis order.

The 518 x 518 input uses DINOv2's native 37 x 37 patch grid, so positional
embedding interpolation is unnecessary.

## Prebuilt package

The validated package is available from the experimental
[`da3-small-coreml-v0.1.0` release](https://github.com/jvcleave/Depth-Anything-3/releases/tag/da3-small-coreml-v0.1.0).
Download both release assets, then verify and unpack them:

```bash
shasum -a 256 -c DepthAnything3SmallCameraTokenImageF16.mlpackage.zip.sha256
unzip DepthAnything3SmallCameraTokenImageF16.mlpackage.zip
```

The archive SHA-256 is
`ffb8150dc55f004bd0bd01f4d1788212c79f46d1af029a88e42e438efa243487`.

## One-command build and validation

Requirements:

- macOS 15 or later
- Python 3.11 available as `python3.11`
- Internet access for the pinned Python packages and public DA3 Small weights
- Approximately 3 GB of free space for the environment, weights, trace, and
  generated package

From the repository root, run:

```bash
tools/coreml/build_da3_small.sh
```

The script creates an isolated `.venv-coreml`, installs the versions in
`requirements-coreml.txt`, downloads `depth-anything/DA3-SMALL` at Hugging Face
revision `e08cab65ca0ec38e7826075418411ab90cab4da3`, verifies the weights SHA-256
`364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`, exports
the model, and validates the Core ML result against the untouched PyTorch model
using `assets/examples/SOH/000.png`.

Core ML Tools 9.0 prints a warning because its published compatibility table
stops at PyTorch 2.7. The pinned PyTorch 2.11 toolchain is intentional: it is the
environment that produced the MESS package and has passed the numerical checks
below. Do not change it based only on that warning.

Artifacts are written to `build/coreml/` and remain outside Git:

```text
build/coreml/DepthAnything3SmallCameraTokenImageF16.mlpackage
build/coreml/DepthAnything3SmallCameraTokenImageF16_traced.pt
```

For repeat builds after the environment has been installed, set
`COREML_SKIP_INSTALL=1`. `PYTHON_BIN`, `COREML_VENV_DIR`, and
`COREML_OUTPUT_DIR` override their corresponding defaults.

## Direct exporter invocation

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv-coreml/bin/python \
  tools/coreml/export_camera_token.py \
  --model-name da3-small \
  --model-source depth-anything/DA3-SMALL \
  --model-revision e08cab65ca0ec38e7826075418411ab90cab4da3 \
  --model-sha256 364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf \
  --input-size 518 \
  --use-image-input \
  --grayscale-output \
  --compute-precision float16 \
  --trace-output build/coreml/DepthAnything3SmallCameraTokenImageF16_traced.pt \
  --output build/coreml/DepthAnything3SmallCameraTokenImageF16.mlpackage
```

The package accepts an RGB image and returns a 518 x 518 grayscale float16
relative-depth image. ImageNet normalization is inside the graph. The deployment
target is iOS 18 / macOS 15 or later.

## Validation recorded on 2026-09-25

- Functional camera-token rewrite versus untouched PyTorch: exact on the fixed
  random input (`max_abs_diff = 0`).
- TorchScript versus untouched PyTorch: `max_abs_diff = 2.38e-7`.
- Float16 Core ML image package versus untouched PyTorch on
  `assets/examples/SOH/000.png`: cosine similarity `1.0`, mean absolute
  difference `0.000542`, maximum absolute difference `0.008437`.
- The older camera-token-disabled image package had Pearson correlation `0.9293`
  against the official output on that sample; this export was effectively `1.0`.
- The generated weight payload is byte-identical to the package integrated into
  MESS.
- Generated package size: approximately 67 MB.

Standalone timing has varied substantially with Core ML runtime state and
machine load. Benchmark the package inside the MESS realtime session before
changing scheduler policy.

Generated `.mlpackage` and TorchScript files are local build artifacts and are
not committed with the exporter.

## Standalone validation

To validate an existing package and collect warm inference timing:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv-coreml/bin/python \
  tools/coreml/validate_export.py \
  --model build/coreml/DepthAnything3SmallCameraTokenImageF16.mlpackage \
  --image assets/examples/SOH/000.png
```

Validation fails if cosine similarity falls below `0.999` or mean absolute
difference exceeds `0.01`. Both thresholds can be overridden on the command
line.

## Current scope

This export is fixed to one view and emits depth only. DA3 Small's confidence
head is still present in the source model but is outside the current MESS depth
contract. MESS must resize its source image to 518 x 518 and scale the returned
relative-depth image back to the source texture size.

The exporter is maintained on top of ByteDance Seed's `main`. The
upstream multi-view reference-selection, batched inference, and streaming
changes do not enter this fixed `B = 1`, `S = 1` graph.
