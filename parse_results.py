"""Parse lm-eval JSON results into a markdown comparison table."""

import glob
import json
import os
import sys

LOG_DIR = "logs"

# Which metric to pull per task (lm-eval metric key format)
METRIC_MAP = {
    "gsm8k_cot": "exact_match,strict-match",
    "arc_easy": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "ifeval": "prompt_level_loose_acc,none",
    "drop": "f1,none",
    "minerva_math": "exact_match,none",
    "mmlu": "acc,none",
}


def parse_logs(log_dir):
    """Return {model_name: {task: score}} from lm-eval JSON results."""
    results = {}
    for model_dir in sorted(glob.glob(os.path.join(log_dir, "*/"))):
        model_name = os.path.basename(model_dir.rstrip("/"))
        # lm-eval nests under a sanitized model name subdir
        json_files = glob.glob(os.path.join(model_dir, "**/results_*.json"), recursive=True)
        if not json_files:
            continue
        # Use the most recent results file
        latest = max(json_files, key=os.path.getmtime)
        try:
            with open(latest) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: failed to read {latest}: {e}", file=sys.stderr)
            continue

        results[model_name] = {}
        for task, task_metrics in data.get("results", {}).items():
            metric_key = METRIC_MAP.get(task)
            if metric_key and metric_key in task_metrics:
                results[model_name][task] = task_metrics[metric_key]
            else:
                # Fall back to first numeric non-stderr metric
                for k, v in task_metrics.items():
                    if isinstance(v, (int, float)) and "stderr" not in k:
                        results[model_name][task] = v
                        break
    return results


def print_table(results):
    # Collect all tasks across all models (preserving order)
    all_tasks = []
    seen = set()
    for scores in results.values():
        for t in scores:
            if t not in seen:
                all_tasks.append(t)
                seen.add(t)

    header = "| Model | " + " | ".join(all_tasks) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_tasks)) + "|"
    print(header)
    print(sep)

    for model, scores in results.items():
        row = f"| {model} |"
        for t in all_tasks:
            if t in scores:
                row += f" {scores[t]:.1%} |"
            else:
                row += " — |"
        print(row)


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else LOG_DIR
    results = parse_logs(log_dir)
    if not results:
        print("No results found.")
    else:
        print_table(results)
