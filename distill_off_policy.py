import argparse
import asyncio
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue as ThreadQueue

import numpy as np
import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from datasets import load_dataset
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from tqdm import tqdm
import math

from transformers import AutoTokenizer

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.allow_unspec_int_on_nn_module = True

DATASET = "allenai/Dolci-Instruct-RL"
TEACHER = "allenai/Olmo-3-7B-Instruct"
STUDENT = "allenai/OLMo-2-0425-1B-Instruct"
HUB_REPO = None  # "hbfreed/Olmo-2-1B-Distilled"
WANDB_PROJECT = "olmo-2-1b-off-policy-distillation"
RUN_NAME = None

STUDENT_DEVICE = "cuda:2"
VLLM_DEVICES = ["cuda:0", "cuda:1"]  # two data-parallel teacher instances

BATCH_SIZE = 1
N_EPOCHS = 1
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 64
MAX_CONTEXT_LENGTH = 2048
LR = 3e-5
MIN_LR_RATIO = 0.1  # decay to 10% of peak LR
MAX_GRAD_NORM = 3.0
WARMUP_STEPS = 50
DECAY_FRACTION = 0.2  # WSD: last 20% is cosine decay
SWEEP_STEPS = None  # set via --sweep to stop early for LR comparison
TEACHER_TOP_K = 128

RESUME_FROM = "checkpoints/offpolicy-olmo3-7b-lr3e-5/latest"
DEBUG_MODE = False

torch.manual_seed(1223)


def parse_args():
    parser = argparse.ArgumentParser(description="Off-policy distillation")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--sweep", type=int, default=None,
                        help="Stop after N optimizer steps (for quick LR sweeps)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Wandb run ID to resume (e.g. xhzvc6kp)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# vLLM worker process — each runs on its own GPU via CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------

def async_vllm_worker(gpu_id, model_name, cmd_q, result_q, ready_event):
    """Async vLLM worker: uses AsyncLLM for continuous batching across steps."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import RequestOutputKind

    engine_args = AsyncEngineArgs(
        model=model_name,
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_logprobs=TEACHER_TOP_K,
    )

    async def run():
        llm = AsyncLLM.from_engine_args(engine_args)
        ready_event.set()

        in_flight = set()

        async def handle_batch(step_idx, token_prompts, params_kwargs):
            """Process one batch: submit all prompts concurrently, gather results."""
            params = SamplingParams(
                **params_kwargs, output_kind=RequestOutputKind.FINAL_ONLY,
            )
            t0 = time.time()

            async def single_request(i, prompt_ids):
                req_id = f"step{step_idx}-prompt{i}"
                final = None
                async for output in llm.generate(
                    {"prompt_token_ids": prompt_ids}, params, req_id,
                ):
                    final = output
                return final

            try:
                outputs = await asyncio.gather(
                    *(single_request(i, p) for i, p in enumerate(token_prompts))
                )
                gen_time = time.time() - t0

                results = []
                for req_output in outputs:
                    prompt_ids = list(req_output.prompt_token_ids)
                    comp = req_output.outputs[0]  # n=1
                    completion_ids = list(comp.token_ids)
                    top_ids_per_pos = []
                    top_lps_per_pos = []
                    for logprob_dict in comp.logprobs:
                        sorted_items = sorted(
                            logprob_dict.items(),
                            key=lambda x: x[1].logprob, reverse=True,
                        )[:TEACHER_TOP_K]
                        top_ids_per_pos.append([t for t, _ in sorted_items])
                        top_lps_per_pos.append([lp.logprob for _, lp in sorted_items])
                    results.append((prompt_ids, completion_ids, top_ids_per_pos, top_lps_per_pos))
                result_q.put((step_idx, results, gen_time))
            except Exception as e:
                print(f"[Worker GPU {gpu_id}] handle_batch error step {step_idx}: {e}")
                result_q.put((step_idx, e, 0.0))

        # Main listener loop
        while True:
            msg = await asyncio.to_thread(cmd_q.get)
            if msg is None:
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
                break
            step_idx, token_prompts, params_kwargs = msg
            task = asyncio.create_task(handle_batch(step_idx, token_prompts, params_kwargs))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        llm.shutdown()
        result_q.put(None)

    asyncio.run(run())



def build_loss_mask(sequences, prompt_lens, pad_token_id):
    """
    Build a mask that's 1.0 for completion tokens, 0.0 for prompt and padding.

    sequences: [batch, seq_len]
    prompt_lens: list[int], length = batch (per-sequence prompt lengths)
    pad_token_id: int

    Returns: [batch, seq_len - 1] (shifted to match logprob indexing)
    """
    batch_size, seq_len = sequences.shape
    positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
    prompt_lens_t = torch.tensor(prompt_lens, device=sequences.device).unsqueeze(1)
    mask = (positions >= prompt_lens_t).float()
    mask[sequences == pad_token_id] = 0.0
    return mask[:, 1:]


SAVE_EVERY = 500  # keep a milestone checkpoint every N steps
CHECKPOINT_BASE = "checkpoints/offpolicy-olmo3-7b-lr3e-5"


def save_checkpoint(student, tokenizer, optimizer, global_step, hub_repo=None):
    """Save rolling 'latest'/'prev' checkpoints, plus a permanent one every SAVE_EVERY steps."""
    import shutil
    latest_dir = f"{CHECKPOINT_BASE}/latest"
    prev_dir = f"{CHECKPOINT_BASE}/prev"

    # Rotate: latest -> prev (so we always have two recent checkpoints)
    if os.path.exists(latest_dir):
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)
        os.rename(latest_dir, prev_dir)

    os.makedirs(latest_dir, exist_ok=True)
    student.save_pretrained(latest_dir)
    tokenizer.save_pretrained(latest_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": global_step},
        f"{latest_dir}/training_state.pt",
    )
    print(f"Saved latest checkpoint (step {global_step}) to {latest_dir}")

    # Keep a permanent copy at milestones
    if global_step % SAVE_EVERY == 0:
        milestone_dir = f"{CHECKPOINT_BASE}/step_{global_step}"
        os.makedirs(milestone_dir, exist_ok=True)
        student.save_pretrained(milestone_dir)
        tokenizer.save_pretrained(milestone_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(), "step": global_step},
            f"{milestone_dir}/training_state.pt",
        )
        print(f"Saved milestone checkpoint to {milestone_dir}")

    if hub_repo:
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=latest_dir,
                repo_id=hub_repo,
                commit_message=f"Step {global_step}",
                ignore_patterns=["training_state.pt"],
            )
            print(f"Pushed checkpoint to {hub_repo}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")


def load_checkpoint(checkpoint_path, student, optimizer):
    """Load optimizer state and return the step to resume from."""
    state_path = f"{checkpoint_path}/training_state.pt"
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        return state["step"]
    return 0


# ---------------------------------------------------------------------------
# Teacher generation via data-parallel vLLM workers
# ---------------------------------------------------------------------------

def build_tensors_from_raw(raw_results, prompt_lens, pad_token_id, vocab_size=None):
    """Convert raw vLLM worker results to padded tensors.

    raw_results: list of (prompt_ids, completion_ids, top_ids_per_pos, top_lps_per_pos)
    Returns: (sequences, teacher_top_ids, teacher_top_lps, attention_mask)
    """
    K = TEACHER_TOP_K
    all_sequences = []
    all_top_ids = []
    all_top_lps = []
    zero_k = [0] * K
    neg_inf_k = [-1e10] * K
    for (prompt_ids, completion_ids, top_ids, top_lps), prompt_len in zip(raw_results, prompt_lens):
        full_seq = prompt_ids + completion_ids
        all_sequences.append(full_seq)
        # Prompt positions get dummy values (masked out by loss mask)
        all_top_ids.append([zero_k] * prompt_len + top_ids)
        all_top_lps.append([neg_inf_k] * prompt_len + top_lps)

    max_seq_len = max(len(seq) for seq in all_sequences)
    for i in range(len(all_sequences)):
        pad_len = max_seq_len - len(all_sequences[i])
        all_sequences[i] += [pad_token_id] * pad_len
        all_top_ids[i] += [zero_k] * pad_len
        all_top_lps[i] += [neg_inf_k] * pad_len

    sequences = torch.from_numpy(np.array(all_sequences, dtype=np.int64))
    if vocab_size is not None:
        sequences[sequences >= vocab_size] = pad_token_id
    attention_mask = (sequences != pad_token_id).long()
    teacher_top_ids = torch.from_numpy(np.array(all_top_ids, dtype=np.int64))   # [batch, seq_len, K]
    teacher_top_lps = torch.from_numpy(np.array(all_top_lps, dtype=np.float32)) # [batch, seq_len, K]

    # Mask out teacher tokens outside student vocab
    if vocab_size is not None:
        oov_mask = teacher_top_ids >= vocab_size
        teacher_top_ids[oov_mask] = 0
        teacher_top_lps[oov_mask] = -1e10

    return sequences, teacher_top_ids, teacher_top_lps, attention_mask


class TeacherPrefetcher:
    """Keeps AsyncLLM workers busy via per-worker sender/receiver threads.

    Each worker has a sender thread (task_q -> cmd_q, fire-and-forget) and a
    receiver thread (result_q -> build tensors -> results dict). The async
    workers run overlapping handle_batch tasks for continuous batching.
    """

    def __init__(self, workers, tokenizer, pad_token_id, max_context,
                 grad_accum_steps, prefetch_depth=6):
        self._workers = workers
        self._tokenizer = tokenizer
        self._pad_token_id = pad_token_id
        self._max_context = max_context
        self._grad_accum_steps = grad_accum_steps
        self._prefetch_depth = prefetch_depth
        self._n_workers = len(workers)

        # Per-worker task queues
        self._task_queues = [ThreadQueue() for _ in range(self._n_workers)]

        # Results storage: step_idx -> (raw_results, prompt_lens, gen_time)
        self._results = {}
        self._cond = threading.Condition()

        # Cache prompt_lens by step_idx for receiver threads
        self._prompt_lens_cache = {}
        self._cache_lock = threading.Lock()

        # Tracking
        self._next_step = 0      # next step the main thread will consume
        self._next_submit = 0    # next step to submit
        self._end_step = 0       # exclusive upper bound for this epoch
        self._all_batches = None

        # Start sender + receiver threads per worker
        self._threads = []
        for worker_idx in range(self._n_workers):
            sender = threading.Thread(
                target=self._sender_loop,
                args=(worker_idx,),
                daemon=True,
            )
            receiver = threading.Thread(
                target=self._receiver_loop,
                args=(worker_idx,),
                daemon=True,
            )
            sender.start()
            receiver.start()
            self._threads.extend([sender, receiver])

    def _build_and_store(self, step_idx, raw_result, prompt_lens, gen_time):
        """Build tensors from raw vLLM output and store for main thread."""
        tensors = build_tensors_from_raw(raw_result, prompt_lens, self._pad_token_id)
        with self._cond:
            self._results[step_idx] = (*tensors, prompt_lens, gen_time)
            self._cond.notify_all()

    def _sender_loop(self, worker_idx):
        """Pull tasks, compute params, send to worker cmd_q."""
        cmd_q = self._workers[worker_idx][0]
        task_q = self._task_queues[worker_idx]

        while True:
            task = task_q.get()
            if task is None:
                cmd_q.put(None)
                break

            step_idx, prompts, prompt_lens = task

            max_prompt_len = max(prompt_lens)
            max_new_tokens = self._max_context - max_prompt_len
            params_kwargs = dict(
                temperature=0.7,
                top_p=1.0,
                max_tokens=max_new_tokens,
                n=1,
                logprobs=TEACHER_TOP_K,
            )

            with self._cache_lock:
                self._prompt_lens_cache[step_idx] = prompt_lens

            cmd_q.put((step_idx, prompts, params_kwargs))

    def _receiver_loop(self, worker_idx):
        """Pull results from worker, build tensors, store for main thread."""
        result_q = self._workers[worker_idx][1]

        while True:
            msg = result_q.get()
            if msg is None:
                break

            step_idx, raw_results, gen_time = msg

            if isinstance(raw_results, Exception):
                with self._cond:
                    self._results[step_idx] = raw_results
                    self._cond.notify_all()
            else:
                with self._cache_lock:
                    prompt_lens = self._prompt_lens_cache.pop(step_idx)
                self._build_and_store(step_idx, raw_results, prompt_lens, gen_time)

    def _prepare_prompts(self, step_idx):
        """Tokenize prompts for one optimizer step."""
        chunk_start = step_idx * self._grad_accum_steps
        chunk_end = chunk_start + self._grad_accum_steps
        raw_prompts = [self._all_batches[i]["prompt"] for i in range(chunk_start, chunk_end)]
        raw_prompts = [p[0] if isinstance(p, list) else p for p in raw_prompts]
        prompts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True
            )
            for p in raw_prompts
        ]
        prompt_lens = [len(p) for p in prompts]
        return prompts, prompt_lens

    def _submit_step(self, step_idx):
        """Submit a step to the appropriate worker's task queue."""
        if step_idx >= self._end_step:
            return
        prompts, prompt_lens = self._prepare_prompts(step_idx)
        worker_idx = step_idx % self._n_workers
        self._task_queues[worker_idx].put((step_idx, prompts, prompt_lens))

    def prime(self, start, end, all_batches):
        """Initialize for an epoch: set batch data and fill prefetch buffer."""
        self._all_batches = all_batches
        self._next_step = start
        self._next_submit = start
        self._end_step = end

        for i in range(start, min(start + self._prefetch_depth, end)):
            self._submit_step(i)
            self._next_submit = i + 1

    def get_next(self):
        """Block until the next sequential result is ready.

        Returns (sequences, teacher_top_ids, teacher_top_lps, attention_mask, prompt_lens, gen_time).
        Also tops up the prefetch buffer.
        """
        step = self._next_step

        # Top up prefetch buffer
        target = min(step + self._prefetch_depth, self._end_step)
        while self._next_submit < target:
            self._submit_step(self._next_submit)
            self._next_submit += 1

        # Wait for result
        with self._cond:
            while step not in self._results:
                self._cond.wait()
            data = self._results.pop(step)

        if isinstance(data, Exception):
            raise data

        self._next_step = step + 1
        return data

    def shutdown(self):
        """Stop sender/receiver threads (senders forward shutdown to workers)."""
        for tq in self._task_queues:
            tq.put(None)
        for t in self._threads:
            t.join(timeout=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    lr = args.lr
    sweep_steps = args.sweep
    wandb_run_id = args.wandb_run_id

    print(f"Loading tokenizer from {TEACHER}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    tokenizer.padding_side = "left"
    PAD_TOKEN_ID = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"Loading dataset from {DATASET}...")
    ds = load_dataset(DATASET, split="train")

    dataset = (
        ds.select_columns(["prompt"])
        .filter(lambda x: len(x["prompt"]) < 2000)
        .shuffle(seed=1223)
    )

    if DEBUG_MODE:
        single = dataset.select(range(1))
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([single] * GRAD_ACCUM_STEPS)
        print(f"DEBUG MODE: 1 prompt repeated {GRAD_ACCUM_STEPS}x for overfitting test")

    n_epochs = 20 if DEBUG_MODE else N_EPOCHS

    batch_size = BATCH_SIZE
    max_context = MAX_CONTEXT_LENGTH

    steps_per_epoch = len(dataset) // (batch_size * GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * n_epochs
    save_every = 50  # save rolling 'latest' every 50 steps; milestones at SAVE_EVERY

    # --- Spawn vLLM teacher workers (must happen before student touches CUDA) ---
    ctx = mp.get_context("spawn")
    workers = []  # list of (input_queue, output_queue)
    processes = []
    ready_events = []
    for device in VLLM_DEVICES:
        gpu_id = int(device.split(":")[1])
        inp_q = ctx.Queue()
        out_q = ctx.Queue()
        ready = ctx.Event()
        p = ctx.Process(
            target=async_vllm_worker,
            args=(gpu_id, TEACHER, inp_q, out_q, ready),
        )
        p.start()
        workers.append((inp_q, out_q))
        processes.append(p)
        ready_events.append(ready)

    # Wait for all workers to load their models
    for i, ready in enumerate(ready_events):
        print(f"Waiting for vLLM teacher on {VLLM_DEVICES[i]}...")
        ready.wait()
    print("All vLLM teacher instances ready")

    # --- Load student model on its own GPU ---
    print(f"Loading student model from {STUDENT}...")
    if RESUME_FROM:
        student = AutoLigerKernelForCausalLM.from_pretrained(
            RESUME_FROM,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(STUDENT_DEVICE)
    else:
        student = AutoLigerKernelForCausalLM.from_pretrained(
            STUDENT, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(STUDENT_DEVICE)

    TEACHER_VOCAB_SIZE = len(tokenizer)
    STUDENT_VOCAB_SIZE = student.config.vocab_size
    if TEACHER_VOCAB_SIZE != STUDENT_VOCAB_SIZE:
        print(f"Vocab mismatch: student={STUDENT_VOCAB_SIZE}, teacher={TEACHER_VOCAB_SIZE}")
        if TEACHER_VOCAB_SIZE > STUDENT_VOCAB_SIZE:
            print(f"Expanding student embeddings {STUDENT_VOCAB_SIZE} -> {TEACHER_VOCAB_SIZE}")
            student.resize_token_embeddings(TEACHER_VOCAB_SIZE)
        else:
            print(f"Student vocab is superset of teacher — no resize needed")
    else:
        print(f"Vocab size: {STUDENT_VOCAB_SIZE}")

    student.gradient_checkpointing_enable()
    student = torch.compile(student)

    optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)
    decay_steps = int(total_steps * DECAY_FRACTION)
    stable_steps = total_steps - warmup_steps - decay_steps

    def wsd_lr_lambda(current_step):
        # Warmup: linear 0 -> 1
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Stable: hold at 1
        if current_step < warmup_steps + stable_steps:
            return 1.0
        # Decay: cosine from 1 -> MIN_LR_RATIO
        progress = (current_step - warmup_steps - stable_steps) / max(1, decay_steps)
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lr_lambda)
    print(f"WSD schedule: {warmup_steps} warmup, {stable_steps} stable, {decay_steps} decay")

    start_step = 0
    if RESUME_FROM:
        start_step = load_checkpoint(RESUME_FROM, student, optimizer)
        # Fast-forward scheduler to match resumed step
        for _ in range(start_step):
            scheduler.step()
        print(f"Resuming from step {start_step}, lr={scheduler.get_last_lr()[0]:.2e}")

    def short_name(model_name: str) -> str:
        return model_name.split("/")[-1]

    run_name = RUN_NAME
    if run_name is None:
        student_short = short_name(STUDENT).lower().replace("olmo-2-0425-", "olmo")
        lr_str = f"{lr:.0e}".replace("-0", "-")
        sweep_tag = f"-sweep{sweep_steps}" if sweep_steps else ""
        run_name = f"{student_short}-offpolicy-distill-lr{lr_str}{sweep_tag}"

    wandb.init(
        project=WANDB_PROJECT,
        id=wandb_run_id,
        name=run_name,
        config={
            "teacher": TEACHER,
            "student": STUDENT,
            "batch_size": batch_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": n_epochs,
            "total_steps": total_steps,
            "lr": lr,
            "max_grad_norm": MAX_GRAD_NORM,
            "warmup_steps": WARMUP_STEPS,
            "max_context_length": MAX_CONTEXT_LENGTH,
            "resume_from": RESUME_FROM,
            "vllm_devices": VLLM_DEVICES,
            "sweep_steps": sweep_steps,
        },
        resume="must" if wandb_run_id else "allow",
    )

    # --- Training loop ---
    global_step = start_step
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    total_optimizer_steps = steps_per_epoch * n_epochs
    pbar = tqdm(total=total_optimizer_steps - start_step, desc="Training")

    torch.cuda.set_device(STUDENT_DEVICE)

    checkpoint_executor = ThreadPoolExecutor(max_workers=1)
    checkpoint_future = None
    PREFETCH_DEPTH = 6
    prefetcher = TeacherPrefetcher(
        workers, tokenizer, PAD_TOKEN_ID, max_context,
        GRAD_ACCUM_STEPS, prefetch_depth=PREFETCH_DEPTH,
    )

    for epoch in range(n_epochs):
        all_batches = list(dataset.iter(batch_size=batch_size))

        # Find the first step we'll actually run
        first_step = max(0, start_step - epoch * steps_per_epoch)

        # Prime the prefetch buffer for this epoch
        prefetcher.prime(first_step, steps_per_epoch, all_batches)

        for opt_step_idx in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + opt_step_idx
            if current_step < start_step:
                continue

            opt_step_start_time = time.time()

            # Get teacher completions (tensors built in feeder thread, overlapped with GPU)
            sequences, teacher_top_ids_all, teacher_top_lps_all, attention_mask, prompt_lens, gen_time = (
                prefetcher.get_next()
            )

            if opt_step_idx < 2:
                print(f"[opt_step {opt_step_idx}] Teacher completion sample: "
                      f"{tokenizer.decode(sequences[0].tolist()[:200])}")

            # Flag sequences that hit max_length without EOS
            positions = torch.arange(sequences.shape[1]).unsqueeze(0)
            prompt_lens_t = torch.tensor(prompt_lens).unsqueeze(1)
            completion_mask = (positions >= prompt_lens_t) & (sequences != PAD_TOKEN_ID)
            hit_eos = ((sequences == tokenizer.eos_token_id) & completion_mask).any(dim=1)

            # Pre-compute loss mask and teacher top-K (shifted to align with student logprobs)
            n_micro_batches = len(sequences) // MICRO_BATCH_SIZE
            loss_mask_all = build_loss_mask(sequences.to(STUDENT_DEVICE), prompt_lens, PAD_TOKEN_ID)
            teacher_top_ids_shifted = teacher_top_ids_all[:, 1:, :].to(STUDENT_DEVICE)
            teacher_top_lps_shifted = teacher_top_lps_all[:, 1:, :].to(STUDENT_DEVICE)
            total_generated_tokens = loss_mask_all.sum().item()

            for mb_idx in range(n_micro_batches):
                seq_start = mb_idx * MICRO_BATCH_SIZE
                seq_end = seq_start + MICRO_BATCH_SIZE

                mb_top_ids = teacher_top_ids_shifted[seq_start:seq_end]
                mb_top_lps = teacher_top_lps_shifted[seq_start:seq_end]
                mb_loss_mask = loss_mask_all[seq_start:seq_end]

                student_input = sequences[seq_start:seq_end].to(STUDENT_DEVICE, non_blocking=True)
                student_mask = attention_mask[seq_start:seq_end].to(STUDENT_DEVICE, non_blocking=True)

                student_out = student(
                    input_ids=student_input,
                    attention_mask=student_mask,
                )

                # Student log-probs over full vocab
                student_log_probs = F.log_softmax(
                    student_out.logits[:, :-1, :], dim=-1
                )

                # Partial KL divergence over teacher's top-K tokens
                student_at_tops = student_log_probs.gather(-1, mb_top_ids)  # [mb, seq-1, K]
                teacher_probs = mb_top_lps.exp()  # [mb, seq-1, K]
                per_token_kl = (teacher_probs * (mb_top_lps - student_at_tops)).sum(dim=-1)  # [mb, seq-1]

                # Student logprob at sampled token (for logging)
                student_lp = student_log_probs.gather(
                    -1, student_input[:, 1:].unsqueeze(-1)
                ).squeeze(-1)

                # Gradient diagnostics (first 3 steps)
                if global_step < 3 and mb_idx == 0:
                    print(f"--- Gradient diagnostics (step {global_step}) ---")
                    print(f"logits.requires_grad: {student_out.logits.requires_grad}")
                    print(f"per_token_kl mean: {per_token_kl.mean().item():.6f}")
                    print(f"loss_mask sum: {mb_loss_mask.sum().item()}")

                # Masked KL loss
                masked_loss = (per_token_kl * mb_loss_mask).sum() / mb_loss_mask.sum()

                if global_step < 3 and mb_idx == 0:
                    print(f"masked_loss: {masked_loss.item():.6f}")
                    print(f"masked_loss.grad_fn: {masked_loss.grad_fn}")
                    print("---")

                scaled_loss = masked_loss / n_micro_batches
                scaled_loss.backward()
                accumulated_loss += scaled_loss.item()

            # --- Optimizer step ---
            if global_step < 3:
                has_grads = any(
                    p.grad is not None for p in student.parameters() if p.requires_grad
                )
                print(f"Step {global_step + 1}: gradients exist = {has_grads}")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), max_norm=MAX_GRAD_NORM
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            opt_step_time = time.time() - opt_step_start_time
            avg_loss = accumulated_loss

            tokens_per_sec = total_generated_tokens / gen_time if gen_time > 0 else 0

            # Compute logging metrics from last micro-batch
            mask_sum = mb_loss_mask.sum()
            mean_student_lp = (student_lp.detach() * mb_loss_mask).sum() / mask_sum
            mean_teacher_lp = (mb_top_lps[:, :, 0] * mb_loss_mask).sum() / mask_sum

            seq_lens_all = attention_mask.to(STUDENT_DEVICE).sum(dim=1)
            prompt_lens_all_t = torch.tensor(prompt_lens, device=STUDENT_DEVICE)
            avg_gen_len = (seq_lens_all - prompt_lens_all_t).float().mean()

            log_payload = {
                "train/loss": avg_loss,
                "train/mean_student_lp": mean_student_lp.item(),
                "train/mean_teacher_lp": mean_teacher_lp.item(),
                "train/grad_norm": grad_norm.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/tokens_per_sec": tokens_per_sec,
                "train/gen_time_sec": gen_time,
                "train/optimizer_step_time_sec": opt_step_time,
                "train/avg_gen_length": avg_gen_len.item(),
                "train/no_eos_frac": (~hit_eos).float().mean().item(),
                "train/global_step": global_step,
            }
            wandb.log(log_payload)

            accumulated_loss = 0.0
            global_step += 1
            pbar.update(1)

            # Early stop for LR sweep
            if sweep_steps and global_step >= sweep_steps:
                print(f"Sweep: stopping after {sweep_steps} steps")
                break

            # Save checkpoint (async)
            hub_repo = None if DEBUG_MODE else HUB_REPO
            if global_step % save_every == 0:
                if checkpoint_future is not None:
                    checkpoint_future.result()
                checkpoint_future = checkpoint_executor.submit(
                    save_checkpoint, student, tokenizer, optimizer, global_step, hub_repo,
                )

        if sweep_steps and global_step >= sweep_steps:
            break

    pbar.close()

    # Shutdown teacher prefetcher
    prefetcher.shutdown()

    if checkpoint_future is not None:
        checkpoint_future.result()
    checkpoint_executor.shutdown(wait=True)

    # Final save
    hub_repo = None if DEBUG_MODE else HUB_REPO
    save_checkpoint(student, tokenizer, optimizer, global_step, hub_repo)

    # Wait for vLLM worker processes to exit
    # (sender threads already sent shutdown sentinels via prefetcher.shutdown())
    for p in processes:
        p.join(timeout=30)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
