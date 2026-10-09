#!/usr/bin/env bash
# Stage 3 shared multitask baseline WITHOUT bridge data (no BiCRR).
# Whole-image letterbox at 640; ring + defect + classification only.
# Usage: bash scripts/run_stage3_nobridge.sh [gpu_id] [resume_checkpoint]
set -euo pipefail

GPU="${1:-1}"
RESUME_CHECKPOINT="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="outputs/stage3_multitask/runs/resnet34_instance_nobridge_v1"
CLASSIFICATION_CHECKPOINT="$PROJECT_ROOT/outputs/classification_baseline/runs/resnet34_balanced/best.pt"
MOKUME_RESTORE="$PROJECT_ROOT/data_processed/bridge_audit/mokume_polygon_restore.json"

if [[ ! -f "$CLASSIFICATION_CHECKPOINT" ]]; then
    echo "Classification checkpoint not found: $CLASSIFICATION_CHECKPOINT" >&2
    exit 1
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume checkpoint not found: $RESUME_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$MOKUME_RESTORE" ]]; then
    echo "Mokume polygon restore mapping not found: $MOKUME_RESTORE" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
EXTRA_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    EXTRA_ARGS+=(--resume "$RESUME_CHECKPOINT" --exist-ok)
fi

exec python scripts/train_stage3_instance_multitask.py \
    --project-root "$PROJECT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --device "$GPU" \
    --backbone resnet34 \
    --anchor-profile thin_crack \
    --imgsz 640 \
    --epochs 100 \
    --steps-per-epoch 256 \
    --ring-batch 4 \
    --detection-batch 2 \
    --classification-batch 16 \
    --task-ratios "ring:2,defect:5,classification:3" \
    --score-weights "ring:1,defect_box:1,defect_mask:1,classification:1" \
    --no-bridge \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --balanced-defect \
    --workers 1 \
    --cpu-threads 4 \
    --eval-interval 5 \
    --cache-dir /tmp/wood_stage3_nobridge_cache \
    --init-classification-checkpoint "$CLASSIFICATION_CHECKPOINT" \
    --mokume-polygon-restore "$MOKUME_RESTORE" \
    "${EXTRA_ARGS[@]}"
