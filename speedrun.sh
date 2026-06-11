#!/usr/bin/env bash

# Exit on error, treat unset variables as error, and fail if any part of a pipe fails
set -euo

START_DIR="$(pwd)"
trap 'cd "$START_DIR"' EXIT  # always return to starting dir

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in $HOME/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat.ltv"

rsync -a --ignore-existing "$HOME/.cache/nanochat.base.240/" "$NANOCHAT_BASE_DIR"

mkdir -p $NANOCHAT_BASE_DIR

export TORCH_LOGS="recompiles"

# -----------------------------------------------------------------------------
# Python venv setup with uv

# # install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# # create a .venv local virtual environment (if it doesn't exist)
# [ -d ".venv" ] || uv venv
# # install the repo dependencies
# uv sync --extra mps

source .venv/bin/activate

# rm -rf rust_ewma/target

cd rust_ewma
uv run maturin build --features python,metal --release
cd ..
uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl


# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

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
# evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
 
# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# Chunk count computation for a depth of 12:
# Original:
# >>> 285_212_672 * 20 * 4.8 / 250_000_000
# 109.521666048
# QKV filter:
# >>> 285544736 * 20 * 4.8 / 250_000_000
# 109.649178624

sleep_if_paused() {
    local pause_file="$1"
    local sleep_index=0

    # turn off -e temporarily inside the function
    set +e

    while [[ -e "$pause_file" ]]; do
        if (( (sleep_index & (sleep_index + 1)) == 0 )); then
            echo "Sleeping at sleep_index: $sleep_index"
        fi
        sleep 1
        ((sleep_index++))
    done

    set -e  # restore -e

    if (( sleep_index != 0 )); then
        echo "Woke up at sleep_index: $sleep_index"
    fi
}

next_pow2() {
    local n=$1
    local p=1
    while (( p < n )); do
        (( p <<= 1 ))
    done
    echo "$p"
}

CONCAT_ORIGINAL_SPEC_11="depth=11 model_type=concat_original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=false ltv_layer_count=0"
REGISTER_CAST_DEPTH_11_LAYER_0_R_128="depth=11 model_type=ltv_fused_concat_register_cast_scan_metal ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true ltv_layer_count=1"
REGISTER_CAST_DEPTH_11_R_128="depth=11 model_type=ltv_fused_concat_register_cast_scan_metal ltv_r=128 ltv_query=active ltv_key=active ltv_value=active alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=true ltv_layer_count=0"
CONCAT_ORIGINAL_SPEC_12="depth=12 model_type=concat_original ltv_r=0 ltv_query=none ltv_key=none ltv_value=none alpha_mlp_bias_init=0 rand_filter_bias=false rand_filter_init=false ltv_layer_count=0"

DEVICE_BATCH_SIZE=4

options=()

options+=("$CONCAT_DUPLICATE_ORIGINAL_SPEC_11")

mid_chat_options=("${options[@]}")

COMBINED_OPTIMIZERS="true"

CORE_METRIC_EVERY=2000

BASE_TRAIN_SAVE_EVERY=60
BASE_TRAIN_NUM_ITERATIONS=10880

# if [[ "$DEPTH" == "12" ]]; then
#     # The original model calculates 10892. Take the value from the QKV filter version.
#     BASE_TRAIN_NUM_ITERATIONS=10880
# elif [[ "$DEPTH" == "1" ]]; then
#     BASE_TRAIN_NUM_ITERATIONS=321
# else
#     BASE_TRAIN_NUM_ITERATIONS=10880
# fi

# SAVE_EVERY=$(( (BASE_TRAIN_SAVE_EVERY * 12 + DEPTH - 1) / DEPTH ))
SAVE_EVERY=60
EXIT_AFTER_SAVE=false

# If EXIT_AFTER_SAVE is true, calculate steps normally.
# Otherwise, we only want to run a single iteration segment.
if [ "$EXIT_AFTER_SAVE" = true ]; then
    num_steps=$(( (BASE_TRAIN_NUM_ITERATIONS + SAVE_EVERY - 1) / SAVE_EVERY ))
else
    num_steps=1
fi

# Save original stdout/stderr
exec 3>&1 4>&2

for ((i=0; i<num_steps; i++)); do
    echo "Step $i / $num_steps"
    sleep_if_paused "/tmp/speedrun.pause"
    for opt in "${options[@]}"; do
        # This seems to be necessary every time.
        cd rust_ewma
        uv run maturin build --features python,metal --release
        cd ..
        uv pip install --force-reinstall --offline rust_ewma/target/wheels/rust_ewma-0.1.1-cp310-cp310-macosx_11_0_arm64.whl

        echo "Running with: $opt"
        eval "$opt"  # sets model_type, ltv_r, etc. as env vars
       
        suffix="d_${depth}-co_${COMBINED_OPTIMIZERS}-mt_${model_type}-r_${ltv_r}-q_${ltv_query}-k_${ltv_key}-v_${ltv_value}-ambi_${alpha_mlp_bias_init}-rfb_${rand_filter_bias}-rfi_${rand_filter_init}"
        (( ltv_layer_count )) && suffix+="-ltv_layer_count_${ltv_layer_count}"

        sleep_if_paused "/tmp/${suffix}.pause"

        logfile="$HOME/tmp/speedrun-$suffix.log"

        exec > >(tee -a "$logfile") 2>&1

        MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/base_train/metal TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/base_train/inductor \
        python -m scripts.base_train \
            --model_tag=${suffix} \
            --report_name=report/${suffix} \
            --depth=$depth \
            --max_seq_len=1024 \
            --run=${WANDB_RUN}-${suffix} \
            --device_batch_size=$DEVICE_BATCH_SIZE \
            --eval_tokens=1310720 \
            --core_metric_every=$CORE_METRIC_EVERY \
            --save_every=$SAVE_EVERY \
            --exit_after_save=$EXIT_AFTER_SAVE \
            --model_dim=1024 \
            --combined_optimizers=${COMBINED_OPTIMIZERS} \
            --model_type=$model_type \
            --ltv_r=$ltv_r \
            --ltv_query=$ltv_query \
            --ltv_key=$ltv_key \
            --ltv_value=$ltv_value \
            --ltv_layer_count=$ltv_layer_count \
            --alpha_mlp_bias_init=$alpha_mlp_bias_init \
            --rand_filter_bias=$rand_filter_bias \
            --rand_filter_init=$rand_filter_init \
            --resume_from_step=-1 \
            --num_iterations=$BASE_TRAIN_NUM_ITERATIONS \
            --sync_on_mps=true

        # Restore stdout/stderr
        exec 1>&3 2>&4
    done
    sleep 1
done

for opt in "${mid_chat_options[@]}"; do
    echo "Running with: $opt"
    eval "$opt"  # sets model_type, ltv_r, etc. as env vars

    suffix="d_${depth}-co_${COMBINED_OPTIMIZERS}-mt_${model_type}-r_${ltv_r}-q_${ltv_query}-k_${ltv_key}-v_${ltv_value}-ambi_${alpha_mlp_bias_init}-rfb_${rand_filter_bias}-rfi_${rand_filter_init}"
    (( ltv_layer_count )) && suffix+="-ltv_layer_count_${ltv_layer_count}"

    logfile="$HOME/tmp/speedrun-$suffix.log"

    exec > >(tee -a "$logfile") 2>&1

    # evaluate the model on a larger chunk of train/val data and draw some samples
    python -m scripts.base_loss \
        --model_tag=${suffix} \
        --report_name=report/${suffix} \
        --device_batch_size=$DEVICE_BATCH_SIZE \
        --split_tokens=1310720 \
        --model_type=$model_type \
        --ltv_query=$ltv_query \
        --ltv_key=$ltv_key \
        --ltv_value=$ltv_value

    # evaluate the model on CORE tasks
    python -m scripts.base_eval \
        --model_tag=${suffix} \
        --report_name=report/${suffix} \
        --model_type=$model_type \
        --ltv_query=$ltv_query \
        --ltv_key=$ltv_key \
        --ltv_value=$ltv_value

    # -----------------------------------------------------------------------------
    # Midtraining (teach the model conversation special tokens, tool use, multiple choice)

    # download 2.3MB of synthetic identity conversations to impart a personality to nanochat
    # see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
    # curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    # run midtraining and eval the model
    MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/mid_train/metal TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/mid_train/inductor \
    python -m scripts.mid_train \
        --model_tag=${suffix} \
        --report_name=report/${suffix} \
        --max_seq_len=1024 \
        --run=${WANDB_RUN}-${suffix}.mid_train \
        --device_batch_size=$DEVICE_BATCH_SIZE \
        --eval_tokens=1310720 \
        --combined_optimizers=${COMBINED_OPTIMIZERS} \
        --model_type=$model_type \
        --ltv_query=$ltv_query \
        --ltv_key=$ltv_key \
        --ltv_value=$ltv_value \
        --sync_on_mps=true

    python -m scripts.chat_eval \
        --model_tag=${suffix} \
        --report_name=report/${suffix} \
        --source mid \
        --batch-size=4 \
        --model_type=$model_type \
        --ltv_query=$ltv_query \
        --ltv_key=$ltv_key \
        --ltv_value=$ltv_value

    # -----------------------------------------------------------------------------
    # Supervised Finetuning (domain adaptation to each sequence all by itself per row)

    # train sft and re-eval right away (should see a small bump)

    MTL_SHADER_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/chat_sft/metal
    TORCHINDUCTOR_CACHE_DIR=$NANOCHAT_BASE_DIR/.training_cache/chat_sft/inductor


    # Restart Threshold.
    # Set high enough to make progress, but low enough to avoid OOM crash.
    PHYSICAL_MEM=$(sysctl -n hw.memsize)
    MEM_THRESHOLD=$(( PHYSICAL_MEM * 3 / 2 ))

    while true; do
        echo "=================================================="
        echo "Starting training process..."
        echo "=================================================="

        set +e  # turn OFF exit-on-error
        MTL_SHADER_CACHE_DIR=$MTL_SHADER_CACHE_DIR \
        TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
        python -m scripts.chat_sft \
            --model_tag=${suffix} \
            --report_name=report/${suffix} \
            --run=${WANDB_RUN}-${suffix}.chat_sft \
            --device_batch_size=$DEVICE_BATCH_SIZE \
            --combined_optimizers=${COMBINED_OPTIMIZERS} \
            --resume_from_step=-1 \
            --save_every=$SAVE_EVERY \
            --restart_memory_threshold=$MEM_THRESHOLD \
            --model_type=$model_type \
            --ltv_query=$ltv_query \
            --ltv_key=$ltv_key \
            --ltv_value=$ltv_value
        EXIT_CODE=$?
        set -e  # turn ON again

        if [ $EXIT_CODE -eq 3 ]; then
            echo "♻️  Process requested restart (Checkpoint Saved/Memory Flush)."
            echo "Restarting in 2 seconds..."
            sleep 2
        else
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ Training finished successfully."
            else
                echo "❌ Training failed with error code $EXIT_CODE."
            fi
            break
        fi
    done

    python -m scripts.chat_eval \
        --model_tag=${suffix} \
        --report_name=report/${suffix} \
        --source sft \
        --batch-size=4 \
        --model_type=$model_type \
        --ltv_query=$ltv_query \
        --ltv_key=$ltv_key \
        --ltv_value=$ltv_value

    # chat with the model over CLI! Leave out the -p to chat interactively
    # python -m scripts.chat_cli -p "Why is the sky blue?"

    # even better, chat with your model over a pretty WebUI ChatGPT style
    # python -m scripts.chat_web

    # -----------------------------------------------------------------------------
    # Reinforcement Learning. Optional, and currently only on GSM8K
    # (optional)

    # run reinforcement learning
    # python -m scripts.chat_rl --run=$WANDB_RUN --combined_optimizers=${COMBINED_OPTIMIZERS}
    # eval the RL model only on GSM8K
    # python -m scripts.chat_eval -i rl -a GSM8K

    # -----------------------------------------------------------------------------
    # Generate the full report by putting together all the sections
    # report.md is the output and will be copied to current directory for convenience
    python -m nanochat.report \
        --report_name=report/${suffix} \
        generate

    exec 1>&3 2>&4
done
