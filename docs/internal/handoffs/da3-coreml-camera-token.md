# Codex Handoff: DA3 Core ML camera-token export

- Updated: 2026-09-25
- Owning repository: `/Users/jvcleave/Documents/WORK_IN_PROGRESS/MACHINE_LEARNING/Depth-Anything-3`
- Branch and HEAD: `mps-benchmarks-etc` at `3a1bb9d`

## Objective

Produce a Core ML `.mlpackage` for DA3 Small that preserves the learned camera
token and alternating attention used by the official single-view model.

## Definition of Done

The functional export graph matches the untouched PyTorch model on fixed inputs,
converts to Core ML, and the Core ML output is checked against the same oracle.

## Current Bounded Milestone

Completed: create and validate a TorchScript/Core ML-compatible functional
replacement for the camera-token mutation without setting `alt_start = -1`.

## Non-Goals

App integration and replacing the production Depth Anything V2 backend.

## Repository State

The repository already contained an untracked conversion script, virtual
environment, four generated `.mlpackage` directories, a traced model, and
`TODO_COREML_CONVERSION.md`. Preserve those artifacts. New work uses a separate
camera-token exporter. `convert_to_coreml_camera_token.py` and
`COREML_CAMERA_TOKEN_EXPORT.md` are the new source artifacts. Generated camera
token model packages and traces remain untracked build artifacts.

## Decisions and Constraints to Preserve

- Preserve `alt_start = 4` for DA3 Small.
- Replace the in-place class/camera-token assignment with concatenation.
- Start at 518 x 518 so DINOv2 positional embeddings require no bicubic resize.
- Compare to the untouched official model, not merely the rewritten eager graph.
- A camera image or camera calibration is not an input; the single-view model
  inserts its learned fixed camera token.

## Relevant Files

- `convert_to_coreml.py` — previous exporter that disabled `alt_start`.
- `convert_to_coreml_camera_token.py` — new fidelity-preserving exporter.
- `COREML_CAMERA_TOKEN_EXPORT.md` — reproduction command, rationale, and results.
- `src/depth_anything_3/model/dinov2/vision_transformer.py` — official camera-token and alternating-attention behavior.

## Verification

- Command: `.venv/bin/python convert_to_coreml_camera_token.py --input-size 518 --use-image-input --grayscale-output --compute-precision float16 ...`
- Latest result: Passed. The functional rewrite is exact against untouched
  PyTorch; trace max error is `2.38e-7`; the float16 image package reached cosine
  `0.9999935` on `assets/examples/SOH/000.png`, measured `23.94 ms` median on
  `CPU_AND_GPU`, and occupies approximately 67 MB.

## Remaining Issues

The generated package is not yet integrated into MESS. Its raw relative-depth
range will need an explicit model scale in the MESS catalog.

## Next Exact Action

Copy or otherwise provision the float16 image package for MESS, add an explicit
518 x 518 model variant and output scale, then test it through the realtime depth
session without changing the V2 production default.

**Fresh-task startup:** Read `docs/internal/handoffs/da3-coreml-camera-token.md`, recover
current state from the repository, and continue with its **Next Exact Action**;
read the listed files first, follow directly referenced files or inspect narrowly
adjacent code as needed, and consult the previous conversation only if the
handoff and repository state lack a required decision, constraint, or
authorization.
