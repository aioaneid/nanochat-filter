#!/usr/bin/env bash

# Exit on error, treat unset variables as error, and fail if any part of a pipe fails
set -eo

START_DIR="$(pwd)"
trap 'cd "$START_DIR"' EXIT  # always return to starting dir

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/cpu_demo_run.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.dev"

mkdir -p "$NANOCHAT_BASE_DIR"

# Danger ahead
rsync -a --delete "$HOME/.cache/nanochat.base.240/" "$NANOCHAT_BASE_DIR"

export TORCH_LOGS=recompiles

# Enable all kernels by default for standard builds
export BUILD_LTV_FUSED_SCAN=1
export BUILD_LTV_FUSED_CONCAT_SCAN=1
export BUILD_LTV_FUSED_CONCAT_HM_SCAN=1
export BUILD_LTV_FUSED_CONCAT_BLELLOCH=1
export BUILD_LTV_FUSED_CONCAT_HM_BLELLOCH=1
export BUILD_LTV_FUSED_CONCAT_HM_ORIGINAL=1
export BUILD_LTV_LOOK_BACK_FUSED_CONCAT_HM_BLELLOCH=1
export BUILD_LTV_PLANE=1

# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# [ -d ".venv" ] || uv venv
# uv sync --extra cpu


cd rust_ewma
uv run maturin build --features python,metal --release
cd ..
uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl

source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# source "$HOME/.cargo/env"
# uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# # Download the first ~2B characters of pretraining dataset
# # look at dev/repackage_data_reference.py for details on how this data was prepared
# # each data shard is ~250M chars
# # so we download 2e9 / 250e6 = 8 data shards at this point
# # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# python -m nanochat.dataset -n 11
# # Immediately also kick off downloading more shards in the background while tokenizer trains
# # See comment below for why 240 is the right number here
# python -m nanochat.dataset -n 240 &
# DATASET_DOWNLOAD_PID=$!
# # train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
# python -m scripts.tok_train --max_chars=2000000000
# # evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

ITER_FACTOR=1

# The eval sequence length seems to grow to 2048 or more.
MAX_SEQ_LEN=128  # 1024
DEVICE_BATCH_SIZE=2  # 4
DEPTH=3  # 12
BASE_TRAIN_ITERATIONS=50  # 50
MID_TRAIN_ITERATIONS=100  # 100
CHAT_SFT_ITERATIONS=4  # 100, but actually 4 because otherwise restarts every 4 iterations

BASE_TRAIN_EVAL_EVERY=${BASE_TRAIN_ITERATIONS}
BASE_TRAIN_CORE_METRIC_EVERY=${BASE_TRAIN_ITERATIONS}
BASE_TRAIN_SAMPLE_EVERY=${BASE_TRAIN_ITERATIONS}

# Restart Threshold.
# Set high enough to make progress, but low enough to avoid OOM crash.
PHYSICAL_MEM=$(sysctl -n hw.memsize)
MEM_THRESHOLD=$(( PHYSICAL_MEM * 3 / 2 ))

# Existing:
values=("active" "none")

# "ltv_scan" "ltv_plane_max" "ltv_shared_pmax"
model_types=("ltv_scan" "ltv_plane_max")

# ADD: ("none" "split_scan" "split_plane_max")
split_values=("split_scan" "split_plane_max")

options+=("model_type=ltv_fused_concat_register_cast_scan_metal ltv_r=1 ltv_query=active ltv_key=active ltv_value=active ltv_layer_count=1")

options+=("model_type=ltv_metal_fused_concat_register_cast_blelloch_scan ltv_r=1 ltv_query=active ltv_key=active ltv_value=active ltv_layer_count=1")

# options+=("model_type=ltv_concat_fused_scan ltv_r=1 ltv_query=active ltv_key=active ltv_value=active ltv_layer_count=1")

ltv_layer_counts=(1 0)

# Existing:
# options+=("model_type=original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none ltv_layer_count=0")

# Existing helper (unchanged):
generate_option() {
    local model_type=$1
    local q=$2
    local k=$3
    local v=$4
    local llc=$5

    # Skip the forbidden combination: true, none, none, none
    if [[ "$q" == "none" && "$k" == "none" && "$v" == "none" ]]; then
        return
    fi

    options+=("model_type=$model_type ltv_r=1 ltv_query=$q ltv_key=$k ltv_value=$v ltv_layer_count=$llc")
}

generate_split_option() {
    local q=$1
    local k=$2
    local v=$3
    local llc=$4

    options+=("model_type=split ltv_r=1 ltv_query=$q ltv_key=$k ltv_value=$v ltv_layer_count=$llc")
}

for q in "${split_values[@]}"; do
    for k in "${split_values[@]}"; do
        for v in "${split_values[@]}"; do
            for llc in "${ltv_layer_counts[@]}"; do
                generate_split_option "$q" "$k" "$v" "$llc"
            done
        done
    done
done

# Existing generation (unchanged):
for mt in "${model_types[@]}"; do
    for q in "${values[@]}"; do
        for k in "${values[@]}"; do
            for v in "${values[@]}"; do
                for llc in "${ltv_layer_counts[@]}"; do
                    generate_option "$mt" "$q" "$k" "$v" "$llc"
                done
            done
        done
    done
done

bools=("true" "false")

# Save original stdout/stderr
exec 3>&1 4>&2

for combined_optimizers in "${bools[@]}"; do
    for opt in "${options[@]}"; do
        echo "Running with: $opt"
        eval "$opt"  # sets model_type, ltv_r, etc. as env vars
        suffix=${combined_optimizers}-${model_type}-${ltv_r}-${ltv_query}-${ltv_key}-${ltv_value}
        (( ltv_layer_count )) && suffix+="-ltv_layer_count_${ltv_layer_count}"

        logfile="/tmp/runcpu-$suffix.log"
        # if [[ -f "$logfile" ]]; then
        #     echo "Skipping existing run: $suffix"
        #     continue
        # fi

        exec > >(tee -a "$logfile") 2>&1
    
        # wipe the report
        python -m nanochat.report --report_name=report/${suffix} reset

        # pretrain the d12 model
        # each optimization step processes a single sequence of 1024 tokens
        # we only run 50 steps of optimization (bump this to get better results)
        MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/base_train/metal TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/base_train/inductor \
        python -m scripts.base_train \
            --depth=$DEPTH \
            --max_seq_len=$MAX_SEQ_LEN \
            --run="dummy" \
            --device_batch_size=$DEVICE_BATCH_SIZE \
            --total_batch_size=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN)) \
            --eval_every=$BASE_TRAIN_EVAL_EVERY \
            --eval_tokens=4096 \
            --core_metric_every=$BASE_TRAIN_CORE_METRIC_EVERY \
            --core_metric_max_per_task=12 \
            --sample_every=$BASE_TRAIN_SAMPLE_EVERY \
            --num_iterations=$((BASE_TRAIN_ITERATIONS * ITER_FACTOR)) \
            --model_dim=1024 \
            --combined_optimizers=$combined_optimizers \
            --model_tag=${suffix} \
            --report_name=report/${suffix} \
            --model_type=$model_type \
            --ltv_r=$ltv_r \
            --ltv_query=$ltv_query \
            --ltv_key=$ltv_key \
            --ltv_value=$ltv_value \
            --ltv_layer_count=$ltv_layer_count \
            --sync_on_mps=true

        python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH_SIZE --split_tokens=4096 \
            --model_tag=${suffix} \
            --report_name=report/${suffix} \
            --model_type=$model_type \
            --ltv_query=$ltv_query \
            --ltv_key=$ltv_key \
            --ltv_value=$ltv_value

        # python -m scripts.base_eval --max-per-task=16 \
        #     --model_tag=${suffix} \
        #     --report_name=report/${suffix} \
        #     --model_type=$model_type \
        #     --ltv_query=$ltv_query \
        #     --ltv_key=$ltv_key \
        #     --ltv_value=$ltv_value

        # # midtraining
        # MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/mid_train/metal TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/mid_train/inductor \
        # python -m scripts.mid_train \
        #     --max_seq_len=$MAX_SEQ_LEN \
        #     --run=${WANDB_RUN}-${suffix} \
        #     --device_batch_size=$DEVICE_BATCH_SIZE \
        #     --eval_every=50 \
        #     --eval_tokens=4096 \
        #     --combined_optimizers=${combined_optimizers} \
        #     --total_batch_size=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN)) \
        #     --num_iterations=$((MID_TRAIN_ITERATIONS * ITER_FACTOR)) \
        #     --model_tag=${suffix} \
        #     --report_name=report/${suffix} \
        #     --model_type=$model_type \
        #     --ltv_query=$ltv_query \
        #     --ltv_key=$ltv_key \
        #     --ltv_value=$ltv_value \
        #     --sync_on_mps=true

        # # eval results will be terrible, this is just to execute the code paths.
        # # note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
        # python -m scripts.chat_eval --source=mid --batch-size=$DEVICE_BATCH_SIZE --max-new-tokens=128 --max-problems=20 \
        #     --model_tag=${suffix} \
        #     --report_name=report/${suffix} \
        #     --model_type=$model_type \
        #     --ltv_query=$ltv_query \
        #     --ltv_key=$ltv_key \
        #     --ltv_value=$ltv_value

        # # -----------------------------------------------------------------------------
        # # Supervised Finetuning (domain adaptation to each sequence all by itself per row)

        # # train sft and re-eval right away (should see a small bump)

        # MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/chat_sft/metal
        # TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/chat_sft/inductor

        # while true; do
        #     echo "=================================================="
        #     echo "Starting training process..."
        #     echo "=================================================="

        #     set +e  # temporarily disable exit-on-error
        #     MTL_SHADER_CACHE_DIR=$MTL_SHADER_CACHE_DIR \
        #     TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
        #     python -m scripts.chat_sft \
        #         --run=${WANDB_RUN}-${suffix} \
        #         --device_batch_size=$DEVICE_BATCH_SIZE \
        #         --combined_optimizers=${combined_optimizers} \
        #         --eval_steps=4 \
        #         --eval_metrics_max_problems=16 \
        #         --num_iterations=$((CHAT_SFT_ITERATIONS * ITER_FACTOR)) \
        #         --resume_from_step=-2 \
        #         --restart_memory_threshold=$MEM_THRESHOLD \
        #         --model_tag=${suffix} \
        #         --report_name=report/${suffix} \
        #         --model_type=$model_type \
        #         --ltv_query=$ltv_query \
        #         --ltv_key=$ltv_key \
        #         --ltv_value=$ltv_value

        #     EXIT_CODE=$?
        #     set -e  # restore exit-on-error

        #     if [ $EXIT_CODE -eq 3 ]; then
        #         echo "♻️  Process requested restart (Checkpoint Saved/Memory Flush)."
        #         echo "Restarting in 1 second..."
        #         sleep 1
        #     else
        #         if [ $EXIT_CODE -eq 0 ]; then
        #             echo "✅ Training finished successfully."
        #         else
        #             echo "❌ Training failed with error code $EXIT_CODE."
        #             exit $EXIT_CODE
        #         fi
        #         break
        #     fi
        # done


        # Chat CLI
        # python -m scripts.chat_cli -p "Why is the sky blue?"

        # Chat Web
        # python -m scripts.chat_web

        python -m nanochat.report --report_name=report/${suffix} generate

        exec 1>&3 2>&4
    done
done
