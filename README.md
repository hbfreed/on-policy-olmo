# On-Policy Distillation: OLMo 2

On-policy distillation of OLMo 2 1B using OLMo 3 7B as the teacher model.

## Goals

1. Validate on-policy distillation pipeline works end-to-end
2. Test distillation with a quantized teacher model

## Approach

Following the on-policy distillation method from [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/) — the student generates outputs, and the teacher provides soft labels on the student's own distribution rather than a fixed dataset.

## Models

| Role    | Model       |
|---------|-------------|
| Student | OLMo 2 1B   |
| Teacher | OLMo 3 7B   |
