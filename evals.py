"""Mid-training evaluation using lm-eval-harness with an existing vLLM engine."""

import os

from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
from transformers import AutoConfig


class VLLMFromExisting(VLLM):
    """Thin wrapper that injects a pre-existing vLLM LLM object into lm-eval's VLLM class.

    Skips the parent __init__ which would load a second copy of the model onto the GPU.
    """

    def __init__(self, llm, tokenizer, model_name):
        # Call grandparent (TemplateLM -> LM) init, skipping VLLM.__init__
        super(VLLM, self).__init__()

        self.model = llm
        self.tokenizer = tokenizer
        self._config = AutoConfig.from_pretrained(model_name)

        # Required VLLM attributes with safe defaults
        self.lora_request = None
        self.data_parallel_size = 1
        self.batch_size = "auto"
        self._max_length = None
        self._max_gen_toks = 256
        self.add_bos_token = None
        self.custom_prefix_token_id = None
        self.tensor_parallel_size = 1
        self.model_args = {"model": model_name}
        self.think_end_token = None
        self.enable_thinking = False
        self.chat_template_args = {}
        self.hf_chat_template = None
        self.truncation_side = "left"
        self.V1 = os.environ.get("VLLM_USE_V1", "1") != "0"


# Tasks that use lm-eval's built-in fewshot defaults
_DEFAULT_TASKS = ["gsm8k_cot", "arc_easy", "ifeval"]
# Tasks requiring explicit num_fewshot override
_FEWSHOT_TASKS = {"truthfulqa_mc2": 6}


def run_evals(vllm_llm, tokenizer, model_name, tasks=None, limit=200):
    """Run lm-eval benchmarks and return a flat dict for wandb.log().

    Runs default tasks in one call, then tasks with custom fewshot in separate calls.
    """
    if tasks is None:
        tasks = _DEFAULT_TASKS + list(_FEWSHOT_TASKS.keys())

    wrapper = VLLMFromExisting(vllm_llm, tokenizer, model_name)
    metrics = {}

    # Split tasks by fewshot config
    default_tasks = [t for t in tasks if t not in _FEWSHOT_TASKS]
    fewshot_tasks = {t: _FEWSHOT_TASKS[t] for t in tasks if t in _FEWSHOT_TASKS}

    if default_tasks:
        results = simple_evaluate(
            model=wrapper,
            tasks=default_tasks,
            limit=limit,
            apply_chat_template=True,
            log_samples=False,
        )
        if results:
            metrics.update(_extract_metrics(results))

    for task, num_fewshot in fewshot_tasks.items():
        results = simple_evaluate(
            model=wrapper,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
            apply_chat_template=True,
            log_samples=False,
        )
        if results:
            metrics.update(_extract_metrics(results))

    return metrics


# Map task -> preferred metric name in lm-eval results
_METRIC_PREFERENCE = {
    "gsm8k_cot": "exact_match,strict-match",
    "arc_easy": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "ifeval": "prompt_level_loose_acc,none",
}


def _extract_metrics(results):
    """Pull per-task scores from lm-eval results into flat wandb-ready dict."""
    metrics = {}
    for task_name, task_results in results.get("results", {}).items():
        # Try preferred metric first, then fall back to first numeric metric
        preferred = _METRIC_PREFERENCE.get(task_name)
        if preferred and preferred in task_results:
            metrics[f"eval/{task_name}"] = task_results[preferred]
        else:
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)) and "stderr" not in metric_name:
                    metrics[f"eval/{task_name}"] = value
                    break
    return metrics
