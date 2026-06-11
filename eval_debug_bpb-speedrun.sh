#!/usr/bin/env bash

exec > >(tee -a $HOME/tmp/speedrun-eval-debug-bpb.log) 2>&1

DRYRUN=false PYTHONUNBUFFERED=1 RUSTUP_INIT_SKIP_PATH_CHECK=yes PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 caffeinate -i time bash -x eval_debug_bpb.sh
