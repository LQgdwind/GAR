# Gradient-Aligned Reward (GAR)

> **Gradients Know What Outcomes Don't: Unlocking Reinforcement Learning for LLM Reasoning with Gradient-Aligned Rewards**

This repository contains the implementation of **Gradient-Aligned Reward (GAR)**, a lightweight online process reward mechanism for reinforcement learning with verifiable rewards (RLVR) in mathematical reasoning.

## Overview

GAR addresses a key limitation of outcome-only reward in RLVR: among correct rollouts, standard verifier rewards provide no signal to distinguish reasoning quality.
GAR computes a gradient-space alignment bonus by:
1. Extracting a gradient-activation product from the LM head for both the model's rollout and an expert Chain-of-Thought anchor
2. Measuring cosine similarity between the two signals
3. Adding a group-centered, clipped bonus on top of the base verifier reward

The gradient signal is obtained via truncated backpropagation through only the output projection layer (lm_head), making it lightweight enough for online RLVR training.

## Repository Structure

```
├── gar/                        # Core GAR implementation
│   ├── online_reward.py        # Main GAR reward computation
│   ├── reward_post_process.py  # Reward post-processing utilities
│   └── __init__.py
├── eval/                       # Public benchmark evaluation queries
│   ├── aime_2026_queries.strict_tags.jsonl
│   ├── hmmt_eval_2025_queries.strict_tags.jsonl
│   ├── hmmt_eval_2026_queries.strict_tags.jsonl
│   └── imo_answerbench_400_queries.strict_tags.jsonl
├── slime_gar.patch             # Patch to apply GAR on top of slime
└── README.md
```

## Reward Formulation

```
if Verify(y) == False:
    reward = 0
else:
    bonus = cos(S_rollout, S_anchor) - mean_group_cosine
    reward = r_base + beta * max(0, bonus)
```

where `S = (dL/dh) * h` is the gradient-activation product at the LM head, and `mean_group_cosine` is the average cosine across all correct rollouts for the same prompt.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gar_base_reward` | 1.0 | Base reward for correct rollouts |
| `gar_beta` | 0.5 | Bonus scaling factor |
| `gar_anchor_aggregation` | `aux_residual` | Aggregation strategy for multiple anchors |
| `gar_aux_weight` | 0.5 | Weight for auxiliary anchors |
| `gar_max_anchors_per_type` | 4 | Max anchors per type (primary / auxiliary) |
| `gar_max_response_tokens` | 16384 | Max tokens for gradient computation |

## Integration with slime

GAR is implemented as a plugin for the [slime](https://github.com/THUDM/slime) RLVR training framework (Megatron backend). The included patch `slime_gar.patch` adds GAR support on top of slime.

### Quick Start

```bash
# 1. Clone slime and checkout the base commit
git clone https://github.com/THUDM/slime.git
cd slime
git checkout 242b073d

# 2. Apply the GAR patch
git apply ../slime_gar.patch

# 3. Install slime following its README

# 4. Launch GAR training
#    Write your own training script following slime's documentation,
#    adding the GAR arguments (see slime/utils/arguments.py for the full list).
#    Example GAR-specific flags:
#      --custom-online-reward-path slime_plugins.gar.online_reward.apply_online_gar
#      --forward-sample-metadata-to-train
#      --gar-ground-truth-key gt_response
#      --gar-anchor-aggregation aux_residual
#      --gar-base-reward 1.0
#      --gar-beta 0.5
```

The patch includes:
- GAR argument definitions (`slime/utils/arguments.py`)
- Online reward hook in the Megatron actor (`slime/backends/megatron_utils/actor.py`)
- Sample metadata forwarding from rollout to actor (`slime/ray/rollout.py`)
- GAR plugin code (`slime_plugins/gar/`)
- Unit tests (`tests/test_gar_*.py`)

## Citation

```bibtex
@inproceedings{gar2026,
  title={Gradients Know What Outcomes Don't: Unlocking Reinforcement Learning for LLM Reasoning with Gradient-Aligned Rewards},
  author={Anonymous},
  year={2026}
}
```

## License

This project is released under the MIT License.
