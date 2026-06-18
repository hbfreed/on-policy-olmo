"""Interactive chat with a local checkpoint via vLLM.

Usage:
    uv run python chat.py [path_to_checkpoint]

Default checkpoint is the distilled-from-base run.
"""
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else \
        "checkpoints/olmo2-7b-to-olmo2-1b-base-revkl/latest"

    print(f"Loading {model} ...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    history = []
    print("\n--- Chat ready. Empty line to exit, '/reset' to clear history. ---\n")

    while True:
        try:
            user = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        if user == "/reset":
            history = []
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        out = llm.generate([prompt], sampling, use_tqdm=False)
        reply = out[0].outputs[0].text
        print(f"\nbot: {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
