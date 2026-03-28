#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="${PROJECT_ROOT}/vendor"

echo "=== Copillust Environment Setup (uv) ==="
echo "Project root: ${PROJECT_ROOT}"

# --- venv ---
if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "Creating venv..."
    uv venv --python 3.12
fi

# --- Core dependencies via pyproject.toml ---
echo "Installing core dependencies..."
uv pip install -e "${PROJECT_ROOT}"

# --- OpenMMLab stack ---
echo "Installing OpenMMLab (mmengine, mmcv, mmdet)..."
uv pip install mmengine "mmcv>=2.0.1,<2.2.0" "mmdet>=3.1.0,<3.4.0"

# --- MMPose (editable from vendor) ---
if [ ! -d "${VENDOR_DIR}/mmpose" ]; then
    echo "Cloning mmpose..."
    mkdir -p "${VENDOR_DIR}"
    git clone --depth 1 https://github.com/open-mmlab/mmpose.git "${VENDOR_DIR}/mmpose"
fi
echo "Installing mmpose (editable)..."
uv pip install -e "${VENDOR_DIR}/mmpose"

# --- Verification ---
echo ""
echo "=== Verification ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"
uv run python -c "
import mmcv, mmengine, mmdet, mmpose
print(f'mmcv: {mmcv.__version__}')
print(f'mmengine: {mmengine.__version__}')
print(f'mmdet: {mmdet.__version__}')
print(f'mmpose: {mmpose.__version__}')
"
uv run python -c "from pose_estimation.core.types import PoseResult; print('pose_estimation package: OK')"

echo ""
echo "=== Setup complete ==="
echo "Run commands with: uv run python ..."
