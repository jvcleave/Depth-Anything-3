# DA3 Small Core ML Camera-Token Export

## Why this exporter exists

The first DA3 Small Core ML experiment set the backbone's `alt_start` to `-1`
because the original camera-token insertion uses an in-place tensor assignment.
That removed both the learned camera token and the alternating global-attention
blocks that follow it. The resulting model converted and ran quickly, but its
output did not reliably behave like coherent scene depth.

`convert_to_coreml_camera_token.py` preserves the official DA3 Small single-view
path. At `S = 1`, the model uses its learned reference camera token; it does not
need camera calibration or a live camera input. Global cross-view attention
naturally reduces to self-attention for the one input view.

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

## MESS-compatible export

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python \
  convert_to_coreml_camera_token.py \
  --model-name da3-small \
  --model-source depth-anything/DA3-SMALL \
  --input-size 518 \
  --use-image-input \
  --grayscale-output \
  --compute-precision float16 \
  --trace-output DepthAnything3SmallCameraTokenImageF16_traced.pt \
  --output DepthAnything3SmallCameraTokenImageF16.mlpackage
```

The package accepts an RGB image and returns a 518 x 518 grayscale float16
relative-depth image. ImageNet normalization is inside the graph. The deployment
target is iOS 18 / macOS 15 or later.

## Validation recorded on 2026-09-25

- Functional camera-token rewrite versus untouched PyTorch: exact on the fixed
  random input (`max_abs_diff = 0`).
- TorchScript versus untouched PyTorch: `max_abs_diff = 2.38e-7`.
- Float16 Core ML image package versus untouched PyTorch on
  `assets/examples/SOH/000.png`: cosine similarity `0.9999935`, mean absolute
  difference `0.000875`, maximum absolute difference `0.008334`.
- The older camera-token-disabled image package had Pearson correlation `0.9293`
  against the official output on that sample; this export was effectively `1.0`.
- Core ML `CPU_AND_GPU`: median `23.94 ms` over 20 warm predictions on the test
  Mac for that sample.
- Generated package size: approximately 67 MB.

Generated `.mlpackage` and TorchScript files are local build artifacts and are
not committed with the exporter.

## Current scope

This export is fixed to one view and emits depth only. DA3 Small's confidence
head is still present in the source model but is outside the current MESS depth
contract. MESS must resize its source image to 518 x 518 and scale the returned
relative-depth image back to the source texture size.
