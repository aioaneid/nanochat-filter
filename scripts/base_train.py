"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os

from enum_actions import enum_action

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

import wandb
import torch
import logging
import pathlib

from nanochat.dataloader import (
    tokenizing_distributed_data_loader,
    tokenizing_distributed_data_loader_with_state,
)
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    get_base_dir,
    autodetect_device_type,
)
from nanochat.gpt_config import UNBOUNDED_LAYER_SPEC, ChunksAndHistory, GPTConfig
from nanochat.utils.tensorboard_adapter import TensorboardAdapter
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.train_utils import sleep_if_paused
from nanochat.checkpoint_manager import (
    save_checkpoint,
    load_checkpoint,
    existing_checkpoints,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

try:
    import rust_ewma
except ImportError:
    rust_ewma = None
from nanochat.gpt_factory import LtvSplitMode, ModelType, one_time_model_factory
from scripts.base_eval import evaluate_model
from distutils.util import strtobool

logger = logging.getLogger(__name__)

# print_banner()

LOG_FACTOR = 1


def parse_bool_array(v):
    # Split by comma and convert each part
    return [bool(strtobool(item.strip())) for item in v.split(",")]


def chunks_and_history_list(value):
    """Parse a string of underscore-separated 'P,Q,K' specs into a list of ChunksAndHistory."""
    specs = []
    for segment in value.split("_"):
        segment = segment.strip()
        if not segment:
            continue
        try:
            p_str, q_str, k_str = segment.split(",")
            p, q, k = int(p_str), int(q_str), int(k_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid layer spec '{segment}': expected format 'P,Q,K' (e.g. '1024,1024,5')"
            )
        specs.append(ChunksAndHistory(p=p, q=q, k=k))
    return specs

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="tensorboard",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument(
    "--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
# Model architecture
parser.add_argument(
    "--depth", type=int, default=20, help="depth of the Transformer model"
)
parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length")
parser.add_argument("--model_dim", type=int, default=0, help="override model dim")
parser.add_argument(
    "--model_type", action=enum_action(ModelType), default=ModelType.ORIGINAL
)
parser.add_argument("--ltv_r", type=int, default=0, help="Enable LTV mode (true/false)")
parser.add_argument(
    "--ltv_query", action=enum_action(LtvSplitMode), default=LtvSplitMode.NONE
)
parser.add_argument(
    "--ltv_key", action=enum_action(LtvSplitMode), default=LtvSplitMode.NONE
)
parser.add_argument(
    "--ltv_value", action=enum_action(LtvSplitMode), default=LtvSplitMode.NONE
)
parser.add_argument(
    "--alpha_mlp_bias_init",
    type=float,
    default=10.0,
    help="Initial value for the alpha MLP bias.",
)
# Training horizon (only one used, in order of precedence)
parser.add_argument(
    "--num_iterations",
    type=int,
    default=-1,
    help="explicit number of optimization steps (-1 = disable)",
)
parser.add_argument(
    "--target_flops",
    type=float,
    default=-1.0,
    help="calculate num_iterations to reach target_flops (-1 = disable)",
)
parser.add_argument(
    "--target_param_data_ratio",
    type=int,
    default=20,
    help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)",
)
parser.add_argument("--sync_on_mps", type=lambda v: bool(strtobool(v)), default=False)
# Optimization
parser.add_argument(
    "--device_batch_size", type=int, default=32, help="per-device batch size"
)
parser.add_argument(
    "--total_batch_size", type=int, default=524288, help="total batch size in tokens"
)
parser.add_argument(
    "--embedding_lr",
    type=float,
    default=0.3,
    help="learning rate for embedding parameters (Adam)",
)
parser.add_argument(
    "--unembedding_lr",
    type=float,
    default=0.004,
    help="learning rate for unembedding parameters (Adam)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="weight decay for embedding/unembedding parameters (Adam)",
)
parser.add_argument(
    "--matrix_lr",
    type=float,
    default=0.02,
    help="learning rate for matrix parameters (Muon)",
)
# Start: Only for the combined optimizer
parser.add_argument(
    "--scalar-lr",
    type=float,
    default=0.5,
    help="learning rate for scalars (resid_lambdas, x0_lambdas)",
)
parser.add_argument(
    "--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding"
)
parser.add_argument(
    "--adam-beta2",
    type=float,
    default=0.95,
    help="Adam beta2 for embedding/unembedding",
)
# End: Only for the combined optimizer
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="gradient clipping value (0.0 = disabled)",
)
parser.add_argument(
    "--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup"
)
parser.add_argument(
    "--warmdown_ratio",
    type=float,
    default=0.4,
    help="ratio of iterations for LR warmdown",
)
parser.add_argument(
    "--final_lr_frac",
    type=float,
    default=0.0,
    help="final LR as fraction of initial LR",
)
parser.add_argument(
    "--resume_from_step",
    type=int,
    default=-1,
    help="resume training from this step (-1 = disable)",
)
parser.add_argument(
    "--combined_optimizers", type=lambda v: bool(strtobool(v)), default=False
)
# Evaluation
parser.add_argument(
    "--eval_every",
    type=int,
    default=250,
    help="evaluate val bpb every N steps (-1 = disable)",
)
parser.add_argument(
    "--eval_tokens",
    type=int,
    default=20 * 524288,
    help="number of tokens to evaluate val loss on",
)
parser.add_argument(
    "--core_metric_every",
    type=int,
    default=2000,
    help="evaluate CORE metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--core_metric_max_per_task",
    type=int,
    default=500,
    help="examples per task for CORE metric",
)
parser.add_argument(
    "--sample_every",
    type=int,
    default=2000,
    help="sample from model every N steps (-1 = disable)",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=-1,
    help="save checkpoints every N steps (-1 = only at end)",
)
parser.add_argument(
    "--save_checkpoint_at_last_step", type=lambda v: bool(strtobool(v)), default=True
)
parser.add_argument(
    "--exit_after_save",
    type=lambda v: bool(strtobool(v)),
    default=False,
)
parser.add_argument(
    "--rand_filter_bias",
    type=lambda v: bool(strtobool(v)),
    default=False,
)
parser.add_argument(
    "--rand_filter_init",
    type=lambda v: bool(strtobool(v)),
    default=True,
)
parser.add_argument(
    "--ltv_layer_count",
    type=int,
    default=0,
    help="number of layers to apply LTV to (if LTV enabled)",
)
parser.add_argument(
    "--layer_specs",
    type=chunks_and_history_list,
    default=[],
    help="Underscore-separated layer specs (P,Q,K). Example: --layer_specs '1024,1024,5_64,60,17_224,100,125'",
)
# Output
parser.add_argument(
    "--model_tag",
    type=str,
    default=None,
    help="override model tag for checkpoint directory name",
)
parser.add_argument("--report_name", type=str, default="report")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type in ["cuda", "mps"]
    else nullcontext()
)
synchronize = (
    torch.cuda.synchronize
    if device_type == "cuda"
    else torch.mps.synchronize
    if device_type == "mps" and args.sync_on_mps
    else lambda: None
)
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else TensorboardAdapter(project="nanochat", name=args.run, config=user_config)
    if args.run.startswith("tensorboard")
    else wandb.init(project="nanochat", name=args.run, config=user_config)
)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = args.depth
model_dim = args.model_dim or (
    args.depth * 64
)  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)


def find_num_heads(model_dim, target_head_dim=128):
    # Find num_heads that divides model_dim evenly, with head_dim closest to target.
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1


num_heads = find_num_heads(model_dim)
num_kv_heads = (
    num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = (
    args.device_batch_size * args.max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)

# -----------------------------------------------------------------------------
# Initialize the Model

model_factory = one_time_model_factory(
    args.model_type, args.ltv_query, args.ltv_key, args.ltv_value
)

if args.ltv_r:
    if args.ltv_layer_count and args.layer_specs:
        assert args.ltv_layer_count == len(args.layer_specs), "Number of layer specs must match ltv_layer_count or be zero"
    if args.layer_specs:
        layer_specs = args.layer_specs
    else:
        layer_specs = [UNBOUNDED_LAYER_SPEC] * (args.ltv_layer_count or num_layers)
else:
    assert args.ltv_layer_count == 0, "ltv_layer_count should be 0 if ltv_r is 0"
    assert not args.layer_specs, "layer_specs should be empty if ltv_r is 0"
    layer_specs = []

# Create a new model with random weights
model_config_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    ltv_r=args.ltv_r,
    ltv_query=args.ltv_query,
    ltv_key=args.ltv_key,
    ltv_value=args.ltv_value,
    layer_specs=layer_specs,
)
# Make a JSON-safe version of the config dict
serializable_config = {**model_config_kwargs}
serializable_config["layer_specs"] = [
    spec.to_dict() for spec in model_config_kwargs["layer_specs"]
]
with torch.device("meta"):
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = GPTConfig(**model_config_kwargs)
    model = model_factory(model_config)
model.to_empty(
    device=device
)  # All tensors get storage on target device but with uninitialized (garbage) data
# All tensors get initialized
model.init_weights(
    alpha_mlp_bias_init=args.alpha_mlp_bias_init,
    rand_filter_bias=args.rand_filter_bias,
    # Repeat the last value as needed
    rand_filter_init=args.rand_filter_init,
)

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
if args.save_checkpoint_at_last_step or args.save_every > 0:
    os.makedirs(checkpoint_dir, exist_ok=True)

# Auto-detect resume step if -2. Otherwise -1 fails if any checkpoints are detected.
if args.resume_from_step == -2:
    steps = existing_checkpoints(checkpoint_dir)
    args.resume_from_step = max(steps, default=-1)
    if args.resume_from_step != -1:
        print0(f"Auto-detected latest checkpoint: {args.resume_from_step}")

resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    if (checkpoint_steps := existing_checkpoints(checkpoint_dir)) and any(
        [cs for cs in checkpoint_steps if cs > args.resume_from_step]
    ):
        raise ValueError(f"Higher checkpoints exist in {checkpoint_steps}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # free up this memory after the copy

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(
    model, dynamic=False, fullgraph=True
)  # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert (
    args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
)
calculated_num_iterations = round(
    args.target_flops / (num_flops_per_token * args.total_batch_size)
)

if args.target_flops > 0:
    # calculate the number of iterations from the target flops
    calculated_num_iterations = round(
        args.target_flops / (num_flops_per_token * args.total_batch_size)
    )
    print0(
        f"Calculated number of iterations from target FLOPs: {calculated_num_iterations:,}"
    )
elif args.target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = args.target_param_data_ratio * num_params
    calculated_num_iterations = target_tokens // args.total_batch_size
    print0(
        f"Calculated number of iterations from target data:param ratio: {calculated_num_iterations:,}"
    )
else:
    raise ValueError("No training horizon specified")

if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(
        f"Using user-provided number of iterations: {num_iterations:,} instead of calculated: {calculated_num_iterations}"
    )
else:
    num_iterations = calculated_num_iterations
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")

total_tokens = args.total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(
    f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_params:.2f}"
)  # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
if args.combined_optimizers:
    # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
    # Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
    batch_lr_scale = 1.0
    reference_batch_size = 2**19
    batch_ratio = args.total_batch_size / reference_batch_size
    if batch_ratio != 1.0:
        # SGD: linear scaling with batch size is standard (not used in nanochat)
        # AdamW: sqrt scaling is standard
        # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
        batch_lr_scale = batch_ratio**0.5
        print0(
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})"
        )

    # Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
    weight_decay_scaled = args.weight_decay * (12 / args.depth) ** 2
    if args.depth != 12:
        print0(
            f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}"
        )

    adam_betas = (args.adam_beta1, args.adam_beta2)
    # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        adam_betas=adam_betas,
        scalar_lr=args.scalar_lr * batch_lr_scale,
    )
else:
    # Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
    optimizers = model.setup_optimizers(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    adamw_optimizer, muon_optimizer = optimizers

if resuming:
    if args.combined_optimizers:
        optimizer.load_state_dict(optimizer_data)
    else:
        for opt, dat in zip(optimizers, optimizer_data):
            opt.load_state_dict(dat)
    del optimizer_data  # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = (
    None if not resuming else meta_data["dataloader_state_dict"]
)
train_loader = tokenizing_distributed_data_loader_with_state(
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    args.device_batch_size, args.max_seq_len, split="val", device=device
)
x, y, dataloader_state_dict = next(
    train_loader
)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers


# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None  # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

pause_file = pathlib.Path("/tmp/base_train.pause")
mfu = 0.0

# -----------------------------------------------------------------------------
# Training loop
while True:
    sleep_if_paused(pause_file)
    last_step = (
        step == num_iterations
    )  # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if saved_checkpoint := (
        (last_step and args.save_checkpoint_at_last_step)
        or (
            step > 0
            and step != args.resume_from_step
            and args.save_every > 0
            and step % max(1, args.save_every // LOG_FACTOR) == 0
        )
    ):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # model parameters
            optimizer.state_dict()
            if args.combined_optimizers
            else [opt.state_dict() for opt in optimizers],  # optimizer states
            {  # metadata saved as json
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": serializable_config,
                "user_config": user_config,  # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {  # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (
        last_step or step % max(1, args.eval_every // LOG_FACTOR) == 0
    ):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (
            args.device_batch_size * args.max_seq_len * ddp_world_size
        )
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        if model_config.ltv_r:
            with torch.no_grad():
                for block in model.transformer.h[: model_config.ltv_layer_count]:
                    if block.attn.qkv_computer.ltv_filter:
                        print0(
                            str(
                                {
                                    name: torch.linalg.vector_norm(p / p.numel()).item()
                                    for name, p in block.attn.qkv_computer.ltv_filter.named_parameters()
                                    if name != "c.weight"
                                }
                                | {
                                    "{}[{}:, :]".format(
                                        name,
                                        start_index := (
                                            model_config.n_head
                                            + 2 * model_config.n_kv_head
                                        )
                                        * model_config.head_dim(),
                                    ): torch.linalg.vector_norm(
                                        (q := p[start_index:, :]) / q.numel()
                                    ).item()
                                    for name, p in block.attn.qkv_computer.ltv_filter.named_parameters()
                                    if name == "c.weight"
                                }
                            )
                        )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (
        last_step
        or (step > 0 and step % max(1, args.core_metric_every // LOG_FACTOR) == 0)
    ):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(
                orig_model,
                tokenizer,
                device,
                max_per_task=args.core_metric_max_per_task,
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if (
        args.sample_every > 0
        and master_process
        and (
            last_step
            or (step > 0 and step % max(1, args.sample_every // LOG_FACTOR) == 0)
        )
    ):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step or (saved_checkpoint and args.exit_after_save):
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # for logging
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
    # gradient clipping
    grad_clip_enabled = args.grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            orig_model.parameters(), args.grad_clip
        )
        grad_norm = (
            grad_norm_tensor.item()
        )  # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    if args.combined_optimizers:
        muon_weight_decay = get_weight_decay(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
    else:
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (
        1 - ema_beta ** (step + 1)
    )  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = (
        989e12 * ddp_world_size
    )  # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""

    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time (s): {total_training_time:.6f}{eta_str}"
    )

    if step % max(1, 100 // LOG_FACTOR) == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # state update
    step += 1

if step >= num_iterations:
    # print a few more stats
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

    # Log to report
    from nanochat.report import get_report

    if args.report_name:
        get_report(args.report_name).log(
            section="Base model training",
            data=[
                user_config,  # CLI args
                {  # stats about the training setup
                    "Number of parameters": num_params,
                    "Number of FLOPs per token": f"{num_flops_per_token:e}",
                    "Calculated number of iterations": num_iterations,
                    "Number of training tokens": total_tokens,
                    "Tokens : Params ratio": args.total_batch_size
                    * num_iterations
                    / num_params,
                    "DDP world size": ddp_world_size,
                    "warmup_ratio": args.warmup_ratio,
                    "warmdown_ratio": args.warmdown_ratio,
                    "final_lr_frac": args.final_lr_frac,
                },
                {  # stats about training outcomes
                    "Minimum validation bpb": min_val_bpb
                    if val_bpb is not None
                    else None,
                    "Final validation bpb": val_bpb,
                    "CORE metric estimate": results.get("core_metric", None),
                    "MFU %": f"{mfu:.2f}%",
                    "Total training flops": f"{flops_so_far:e}",
                    "Total training time": f"{total_training_time / 60:.2f}m",
                    "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
                },
            ],
        )

    # cleanup
    wandb_run.finish()  # wandb run finish
    compute_cleanup()
