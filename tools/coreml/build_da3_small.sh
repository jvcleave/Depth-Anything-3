#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${COREML_VENV_DIR:-$REPO_ROOT/.venv-coreml}"
OUTPUT_DIR="${COREML_OUTPUT_DIR:-$REPO_ROOT/build/coreml}"
MODEL_BASENAME="DepthAnything3SmallCameraTokenImageF16"
MODEL_PATH="$OUTPUT_DIR/$MODEL_BASENAME.mlpackage"
TRACE_PATH="$OUTPUT_DIR/${MODEL_BASENAME}_traced.pt"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python 3.11 is required. Set PYTHON_BIN to its executable." >&2
    exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "The tested conversion environment requires Python 3.11; found $PYTHON_VERSION." >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ "${COREML_SKIP_INSTALL:-0}" != "1" ]]; then
    "$VENV_DIR/bin/python" -m pip install "pip==26.2.1"
    "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-coreml.txt"
fi

mkdir -p "$OUTPUT_DIR"

KMP_DUPLICATE_LIB_OK=TRUE "$VENV_DIR/bin/python" \
    "$SCRIPT_DIR/export_camera_token.py" \
    --model-name da3-small \
    --model-source depth-anything/DA3-SMALL \
    --model-revision e08cab65ca0ec38e7826075418411ab90cab4da3 \
    --model-sha256 364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf \
    --input-size 518 \
    --use-image-input \
    --grayscale-output \
    --compute-precision float16 \
    --trace-output "$TRACE_PATH" \
    --output "$MODEL_PATH"

KMP_DUPLICATE_LIB_OK=TRUE "$VENV_DIR/bin/python" \
    "$SCRIPT_DIR/validate_export.py" \
    --model "$MODEL_PATH" \
    --image "$REPO_ROOT/assets/examples/SOH/000.png" \
    --model-source depth-anything/DA3-SMALL \
    --model-revision e08cab65ca0ec38e7826075418411ab90cab4da3 \
    --model-sha256 364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf

echo "Core ML package: $MODEL_PATH"
echo "TorchScript trace: $TRACE_PATH"
