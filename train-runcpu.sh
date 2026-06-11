#!/usr/bin/env bash

exec > >(tee -a $HOME/tmp/runcpu-ltv.log) 2>&1

PYTHONUNBUFFERED=1 RUSTUP_INIT_SKIP_PATH_CHECK=yes WANDB_RUN=tensorboard PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 caffeinate -i time bash -x dev/runcpu.sh
