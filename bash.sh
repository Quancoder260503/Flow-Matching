#!/bin/bash

SESSION="otflow_train"

if [ -n "$TMUX" ]; then
    echo "Already inside tmux — running directly."
elif tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '${SESSION}' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
else
    echo "Starting new tmux session: ${SESSION}"
    tmux new-session -d -s "$SESSION" "bash $0; read -p 'Done. Press Enter to close.'"
    echo "Training running in background."
    echo "  Attach:  tmux attach -t ${SESSION}"
    echo "  Detach:  Ctrl+B then D"
    tmux attach -t "$SESSION"
    exit 0
fi

set -e

# --- GPU Config ---
# Your launch() calls mp.spawn internally — just set which GPUs are visible
CUDA_VISIBLE_DEVICES="0,1"

# --- Model Config ---
MODEL_VERSION="OTFlowMatchingCIFAR10_64x64"
TRAIN_MODE="ContOtFlowMatching"
DOC="ContOTFlowMatchingCIFAR10_64x64"
RESUME_TRAINING="False"
MODEL_CHANNELS=96
IMAGE_SIZE=64
IN_CHANNELS=3
OUT_CHANNELS=3
SAMPLER="heun"
DATASET="CIFAR10"
SKEWED_TIMESTEPS="True"
TRAIN="True"

# --- Run ---
echo "============================================================"
echo "  Starting OTFlowMatching Training"
echo "  Model:    ${MODEL_VERSION}"
echo "  Dataset:  ${DATASET} (${IMAGE_SIZE}x${IMAGE_SIZE})"
echo "  Sampler:  ${SAMPLER}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "  Doc:      ${DOC}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python3 Main.py \
    --model_version "${MODEL_VERSION}" \
    --train_mode "${TRAIN_MODE}" \
    --doc "${DOC}" \
    --resume_training "${RESUME_TRAINING}" \
    --model_channels "${MODEL_CHANNELS}" \
    --image_size "${IMAGE_SIZE}" \
    --in_channels "${IN_CHANNELS}" \
    --out_channels "${OUT_CHANNELS}" \
    --sampler "${SAMPLER}" \
    --dataset "${DATASET}" \
    --skewed_timesteps "${SKEWED_TIMESTEPS}" \
    --train "${TRAIN}"

echo "============================================================"
echo "  Training complete!"
echo "============================================================"