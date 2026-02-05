---
license: apache-2.0
base_model:
  - allenai/OLMo-2-0425-1B-Instruct
  - allenai/Olmo-3-7B-Think
tags:
  - distillation
  - reasoning
  - olmo
---

# OLMo-2-1B-Distilled

A 1B parameter model trained via **on-policy distillation** to produce reasoning traces.

## Training Approach

1. **SFT Stage**: Fine-tune OLMo-2-0425-1B-Instruct on examples with `<think>...</think>` reasoning format
2. **On-Policy Distillation**: Student generates rollouts, teacher (OLMo-3-7B-Think) provides token-level supervision via JSD loss

## Models

- **Student**: [allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)
- **Teacher**: [allenai/Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think)

## Output Format

The model produces reasoning in the format:
```
<think>
[internal reasoning]
</think>
[final answer]
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("hbfreed/Olmo-2-1B-Distilled")
tokenizer = AutoTokenizer.from_pretrained("hbfreed/Olmo-2-1B-Distilled")
```
