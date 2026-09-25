# Codex Handoff: DA3 Core ML camera-token export

- Updated: 2026-09-25
- Owning repository: `/Users/jvcleave/Documents/WORK_IN_PROGRESS/MACHINE_LEARNING/Depth-Anything-3`
- Integration worktree: `/Users/jvcleave/Documents/WORK_IN_PROGRESS/MACHINE_LEARNING/Depth-Anything-3-coreml-upstream`
- Branch: `codex/coreml-upstream`; recover the current revision with `git log`

## Objective

Maintain a reproducible Core ML `.mlpackage` build for DA3 Small that preserves
the learned camera token and alternating attention used by the official
single-view model.

## Definition of Done

A clean Python 3.11 environment can run one command to export the model, compare
it with untouched PyTorch, and validate the resulting Core ML package.

## Current Bounded Milestone

Completed: port the validated exporter to ByteDance Seed `main` at `3d835ec`,
pin the Hugging Face model revision and checksum, then rebuild and validate it.

## Non-Goals

Replacing the production Depth Anything V2 default or committing generated
model artifacts to Git.

## Repository State

The repository already contained an untracked conversion script, virtual
environment, four generated `.mlpackage` directories, a traced model, and
`TODO_COREML_CONVERSION.md`. Preserve those artifacts. The supported camera-token
conversion workflow now lives under `tools/coreml/`. Generated model packages
and traces remain ignored build artifacts.

## Decisions and Constraints to Preserve

- Preserve `alt_start = 4` for DA3 Small.
- Replace the in-place class/camera-token assignment with concatenation.
- Start at 518 x 518 so DINOv2 positional embeddings require no bicubic resize.
- Compare to the untouched official model, not merely the rewritten eager graph.
- A camera image or camera calibration is not an input; the single-view model
  inserts its learned fixed camera token.
- Pin DA3-SMALL to Hugging Face revision
  `e08cab65ca0ec38e7826075418411ab90cab4da3` and verify weights SHA-256
  `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.

## Relevant Files

- `convert_to_coreml.py` — previous exporter that disabled `alt_start`.
- `tools/coreml/export_camera_token.py` — fidelity-preserving exporter.
- `tools/coreml/build_da3_small.sh` — isolated one-command build and validation.
- `tools/coreml/validate_export.py` — Core ML comparison and timing.
- `tools/coreml/README.md` — reproduction command, rationale, and results.
- `src/depth_anything_3/model/dinov2/vision_transformer.py` — official camera-token and alternating-attention behavior.

## Verification

- Command: `tools/coreml/build_da3_small.sh`
- Latest current-upstream result: Passed. The functional rewrite is exact
  against untouched PyTorch; trace max error is `2.38e-7`; the float16 image
  package reached cosine `1.0`, mean absolute difference `0.000542`, and maximum
  absolute difference `0.008437` on `assets/examples/SOH/000.png`. Its weight
  payload is byte-identical to the v0.1 release and package integrated into
  MESS. Runtime timing varies between standalone runs and should be measured in
  the MESS session for scheduling decisions.

## Remaining Issues

The package is published in the experimental `da3-small-coreml-v0.1.0` GitHub
release with a separate SHA-256 asset. The fork's default branch still needs a
small README pointer to this clean integration branch. Realtime MESS performance
still needs to be compared against V2 under the same session workload.

## Next Exact Action

Push `codex/coreml-upstream`, then add a README pointer on the fork's default
`main` branch without merging its older MPS-specific source changes into the
clean upstream-based branch.

**Fresh-task startup:** Read `docs/internal/handoffs/da3-coreml-camera-token.md`, recover
current state from the repository, and continue with its **Next Exact Action**;
read the listed files first, follow directly referenced files or inspect narrowly
adjacent code as needed, and consult the previous conversation only if the
handoff and repository state lack a required decision, constraint, or
authorization.
