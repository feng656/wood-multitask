#!/bin/bash
# Stage 3: Shared Multi-Task Model (no BiCRR)
# Run from server: bash scripts/run_stage3_instance.sh [gpu_id]
set -euo pipefail
GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Stage 3: Shared Instance Multi-Task (no BiCRR) ==="
echo "GPU: $GPU"
echo "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

python scripts/train_stage3_instance_multitask.py \
    --device "$GPU" \
    --backbone resnet34 \
    --imgsz 320 \
    --epochs 100 \
    --steps-per-epoch 512 \
    --ring-batch 16 \
    --detection-batch 4 \
    --classification-batch 64 \
    --bridge-batch 2 \
    --task-ratios "ring:1,defect:6,classification:4,bridge:1" \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --workers 4 \
    --cpu-threads 8 \
    --eval-interval 10 \
    --cache-dir /tmp/wood_stage3_cache
