"""Standalone GSM8K eval via vLLM — replaces the lm-eval path (incompatible with vllm 0.23).

Greedy chain-of-thought generation, extract the final number, score vs the gold answer.
Run each model in its own process so the GPU frees between models:

    CUDA_VISIBLE_DEVICES=0 uv run python eval_gsm8k.py <model> [--n 200] [--max-tokens 768]
"""
import argparse
import re

INSTR = "\n\nPlease reason step by step, then give your final answer after '####'."


def extract_answer(text):
    """Final number after '####' if present, else the last number anywhere."""
    seg = text.rsplit("####", 1)[-1] if "####" in text else text
    nums = re.findall(r"-?\$?[\d,]*\.?\d+", seg) or re.findall(r"-?\$?[\d,]*\.?\d+", text)
    if not nums:
        return None
    return nums[-1].replace("$", "").replace(",", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.90)

    ds = load_dataset("gsm8k", "main", split="test").select(range(args.n))
    prompts, golds = [], []
    for ex in ds:
        q = ex["question"] + INSTR
        if tok.chat_template:  # instruct models (distilled / teacher)
            p = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
        else:  # base model: plain prompt
            p = q + "\n"
        prompts.append(p)
        golds.append(ex["answer"].rsplit("####", 1)[-1].strip().replace(",", ""))

    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    correct = 0
    for o, gold in zip(outs, golds):
        pred = extract_answer(o.outputs[0].text)
        try:
            if pred is not None and abs(float(pred) - float(gold)) < 1e-4:
                correct += 1
        except ValueError:
            pass
    acc = correct / len(golds)
    print(f"RESULT model={args.model} n={len(golds)} gsm8k={acc:.3f} ({correct}/{len(golds)})")


if __name__ == "__main__":
    main()
