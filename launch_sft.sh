#!/usr/bin/env bash
# Launch SFT / off-policy distillation with DDP + sequence packing.
# Prepares data first (CPU only), waits for GPUs to free up, then trains.
#
# Usage: uv run bash launch_sft.sh {kld|cce} [extra args...]
#
# Examples:
#   uv run bash launch_sft.sh cce --sweep 3 --total-batch-size 128 --pack-length 2048
#   uv run bash launch_sft.sh kld --sweep 3 --total-batch-size 128 --pack-length 2048

set -e

LOSS_TYPE=${1:-kld}
shift

# Extract --pack-length from args (default 2048) for the prep step
PACK_LENGTH=2048
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--pack-length" ]]; then
        PACK_LENGTH="${ARGS[$((i+1))]}"
        break
    fi
done

# Stage 1: Prepare data (CPU only, no GPU needed)
echo "=== Preparing data (pack_length=$PACK_LENGTH) ==="
python prepare_sft_data.py --pack-length "$PACK_LENGTH"

# Stage 2: Wait for GPUs to be free
if [ "$LOSS_TYPE" = "kld" ]; then
    NPROC=3
    NEEDED_GPUS="0,1,2"  # 3 DDP students, teacher logprobs pre-extracted
else
    NPROC=3
    NEEDED_GPUS="0,1,2"
fi

echo "=== Waiting for GPUs ($NEEDED_GPUS) to be free ==="
while true; do
    # Check if any of the needed GPUs have processes running
    BUSY=0
    for gpu_id in ${NEEDED_GPUS//,/ }; do
        PROCS=$(nvidia-smi --id=$gpu_id --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
        if [ "$PROCS" -gt 0 ]; then
            BUSY=1
            break
        fi
    done
    if [ "$BUSY" -eq 0 ]; then
        echo "GPUs are free!"
        break
    fi
    echo "  GPUs busy, waiting 30s..."
    sleep 200 
done

# Stage 3: Launch training
echo "=== Launching SFT ($LOSS_TYPE, $NPROC ranks) ==="
torchrun --nproc_per_node=$NPROC sft_ddp.py --loss-type "$LOSS_TYPE" "$@"
