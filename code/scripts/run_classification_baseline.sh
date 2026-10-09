#!/bin/bash
# Classification baseline training — ResNet34 3-class (normal/knot/crack)
# Run from server: bash scripts/run_classification_baseline.sh [gpu_id]

set -euo pipefail
GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Classification Baseline ==="
echo "GPU: $GPU"
echo "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

python scripts/train_classification_baseline.py \
    --device "$GPU" \
    --model resnet34 \
    --epochs 30 \
    --batch 64 \
    --lr 3e-4 \
    --imgsz 224 \
    --workers 4 \
    --cpu-threads 8 \
    --eval-splits val test external_test
