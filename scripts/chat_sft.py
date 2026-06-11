"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import argparse
import gc
import os
import sys
import logging
from enum_actions import enum_action
import wandb
import pathlib
import torch
import torch.distributed as dist
from contextlib import nullcontext
from distutils.util import strtobool

try:
    import rust_ewma
except ImportError:
    rust_ewma = None
from nanochat.gpt_factory import LtvSplitMode, ModelType, one_time_model_factory

# Ensure memory fragmentation doesn't crash us immediately
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint, load_checkpoint, existing_checkpoints
from nanochat.engine import Engine
from nanochat.utils.tensorboard_adapter import TensorboardAdapter
from nanochat.train_utils import sleep_if_paused
from scripts.chat_eval import run_chat_eval
from nanochat.report import get_report

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

logger = logging.getLogger(__name__)

LOG_FACTOR = 1

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised finetuning for chat")
# Logging
parser.add_argument("--run", type=str, default="tensorboard", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="mid", help="base|mid - which checkpoint to load from")
parser.add_argument("--model_tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model_step", type=int, default=None, help="model step to load from")
parser.add_argument('--model_type', action=enum_action(ModelType), default=ModelType.ORIGINAL)
parser.add_argument('--ltv_query', action=enum_action(LtvSplitMode), default=LtvSplitMode.NONE)
parser.add_argument('--ltv_key', action=enum_action(LtvSplitMode), default=LtvSplitMode.NONE)
parser.add_argument('--ltv_value', action=enum_action(LtvSplitMode), default=LtvSplitMode.NONE)
# Resuming & Restarting
parser.add_argument("--resume_from_step", type=int, default=-1, help="step to resume from (-2 = auto-detect; -1 = do not resume)")
parser.add_argument("--save_every", type=int, default=0, help="save checkpoint and exit every N steps (0=disabled)")
parser.add_argument("--restart_memory_threshold", type=int, default=sys.maxsize, help="restart if MPS driver memory exceeds this (in GB)")
# Training horizon
parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--num_iterations", type=int, default=-1, help="override number of iterations (-1 = use num_epochs)")
# Batch sizes
parser.add_argument("--device_batch_size", type=int, default=4, help="per-device batch size")
parser.add_argument("--target_examples_per_step", type=int, default=32, help="target examples per optimization step")
# Optimization
parser.add_argument("--embedding_lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding_lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init_lr_frac", type=float, default=0.02, help="initial LR as fraction of base LR")
parser.add_argument("--combined_optimizers", type=lambda v: bool(strtobool(v)), default=False)
# Evaluation
parser.add_argument("--eval_every", type=int, default=100, help="evaluate val loss every N steps")
parser.add_argument("--eval_steps", type=int, default=100, help="number of batches for val loss evaluation")
parser.add_argument("--eval_metrics_every", type=int, default=200, help="evaluate accuracy metrics every N steps")
parser.add_argument("--eval_metrics_max_problems", type=int, default=1024, help="max problems per metric evaluation")
# Output
parser.add_argument("--report_name", type=str, default="report")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type in ["cuda", "mps"] else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else TensorboardAdapter(project="nanochat-sft", name=args.run, config=user_config, save_code=True) if args.run.startswith("tensorboard") else wandb.init(project="nanochat-sft", name=args.run, config=user_config, save_code=True)

# -----------------------------------------------------------------------------
# 1. Initialize Model & Tokenizer
# -----------------------------------------------------------------------------
# Load the base model first (we might overwrite weights later if resuming)
model, tokenizer, meta = load_model(
    one_time_model_factory(
        args.model_type, args.ltv_query, args.ltv_key, args.ltv_value
    ),
    args.source,
    device,
    phase="train",
    model_tag=args.model_tag,
    step=args.model_step,
)
orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # avoiding compile on MPS for now due to memory issues
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# 2. Checkpoint Directory & Resume Logic
# -----------------------------------------------------------------------------
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else "chat_sft"
checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
os.makedirs(checkpoint_dir, exist_ok=True)

# Auto-detect resume step if -2. Otherwise -1 fails if any checkpoints are detected.
if args.resume_from_step == -2:
    steps = existing_checkpoints(checkpoint_dir)
    args.resume_from_step = max(steps, default=-1)
    if args.resume_from_step != -1:
        print0(f"Auto-detected latest checkpoint: {args.resume_from_step}")

# -----------------------------------------------------------------------------
# 3. Data Loaders
# -----------------------------------------------------------------------------
# Task data mixture we'll train on
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(tokenizer=tokenizer, split="train", stop=10_000), # 10K rows of smoltalk
    CustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
    SimpleSpelling(size=300, split="train"), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=300, split="train"), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
val_ds = SmolTalk(tokenizer=tokenizer, split="test") # general conversations, 24K rows (though we don't actually use all of it)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if args.num_iterations == -1:
    # derive num_iterations from num_epochs and the size of the dataset
    assert args.num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // args.target_examples_per_step) * args.num_epochs
else:
    num_iterations = args.num_iterations
train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

if args.combined_optimizers:
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]
else:    
    optimizers = model.setup_optimizers(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    # Set the initial learning rate as a fraction of the base learning rate
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * args.init_lr_frac
            group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# 5. Load Checkpoint (if resuming)
# -----------------------------------------------------------------------------
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    if (checkpoint_steps := existing_checkpoints(checkpoint_dir)) and any(
        [cs for cs in checkpoint_steps if cs > args.resume_from_step]
    ):
        raise ValueError(f"Higher checkpoints exist in {checkpoint_steps}")
    
    # Load checkpoint data
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, 
        args.resume_from_step, 
        device, 
        load_optimizer=True, 
        rank=ddp_rank
    )
    
    # Apply model weights
    model.load_state_dict(model_data, strict=False) # strict=False for SFT safety
    del model_data
    
    # Apply optimizer states (Always restore)
    if optimizer_data is not None:
        if args.combined_optimizers:
            optimizer.load_state_dict(optimizer_data)
        else:
            # Assuming optimizers list order matches saved list order
            for opt, state in zip(optimizers, optimizer_data):
                opt.load_state_dict(state)
    del optimizer_data
    
    print0("Checkpoint loaded successfully.")

# -----------------------------------------------------------------------------
# 6. Training Loop Setup
# -----------------------------------------------------------------------------
# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

def free_up_memory():
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    torch.mps.empty_cache()

# Fast-forward loader
iter_start = args.resume_from_step if resuming else 0
if iter_start > 0:
    print0(f"Fast-forwarding data loader by {iter_start} steps...")
    for _ in range(iter_start * grad_accum_steps):
        next(train_loader)

pause_file = pathlib.Path("/tmp/chat_sft.pause")

# Go!
metrics = {}
for step in range(iter_start, num_iterations):
    sleep_if_paused(pause_file)
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if args.eval_every > 0 and (last_step or (step and step % max(1, args.eval_every // LOG_FACTOR) == 0)):
        model.eval()
        val_loader = build_val_loader()
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # evaluate accuracy of the multiple choice tasks (which are quick to run)
    if args.eval_metrics_every > 0 and (last_step or (step and step % max(1, args.eval_metrics_every // LOG_FACTOR) == 0)):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=args.device_batch_size*2, max_problems=args.eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=args.device_batch_size*2, max_problems=args.eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        logger.debug("train_inputs.shape: %s", train_inputs.shape)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        # Based on mid_train, add a synchronization point here. No idea if needed or if it affects training speed.
        train_loss_item = loss.item()
        logger.debug("micro_step: %d loss: %s", micro_step, train_loss_item)
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    if args.combined_optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        # step the optimizer
        optimizer.step()            
    else:
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        # step the optimizers
        for opt in optimizers:
            opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    print0(f"MPS allocated: {torch.mps.current_allocated_memory()} | MPS driver: {torch.mps.driver_allocated_memory()}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })

    # -------------------------------------------------------------------------
    # Checkpoint & Restart Logic
    # -------------------------------------------------------------------------
    should_save = False
    reason = ""

    # Condition A: Memory Pressure on MPS
    if device.type == "mps":
        current_mem = torch.mps.driver_allocated_memory()
        if (old_mem := current_mem) > args.restart_memory_threshold:
            free_up_memory()
            current_mem = torch.mps.driver_allocated_memory()
            logger.info("Emptied cache because memory too high. Before: %s After: %s", old_mem, current_mem)
        if current_mem > args.restart_memory_threshold:
            should_save = True
            reason = f"memory limit ({current_mem} > {args.restart_memory_threshold})"
            print0("⚠️ Memory too high")

    # Condition B: Standard save_every
    if args.save_every > 0 and (step + 1) % args.save_every == 0:
        should_save = True
        reason = "scheduled"

    # Condition C: Last step
    if last_step:
        should_save = True
        reason = "last step"

    if should_save:
        print0(f"Triggering save due to: {reason}")
        if master_process:
            opt_state = optimizer.state_dict() if args.combined_optimizers else [opt.state_dict() for opt in optimizers]
            save_checkpoint(
                checkpoint_dir,
                step + 1, # Save as the next step to start from
                model.state_dict(),
                opt_state, # Save optimizer state
                {
                    "step": step + 1,
                    "val_loss": val_loss if 'val_loss' in locals() else 0.0,
                    **metrics,
                    "model_config": model.config.__dict__  # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
                }
            )
        if ddp:
            dist.barrier()
        if "memory" in reason:
            print0(f"⚠️ Exiting with code 3 (restart request) due to: {reason}")
            sys.exit(3)


# Final Save
print(f"✅ Finished")

# Log to report
get_report(args.report_name).log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
