import os
import re

import torch

from slime.utils.processing_utils import load_tokenizer


STRICT_FORMAT_TAGS = ("<|think|>", "<|/think|>", "<answer>", "</answer>")
STRICT_FORMAT_SPECIAL_TOKEN_TAGS = {tag for tag in STRICT_FORMAT_TAGS if tag.startswith("<|") and tag.endswith("|>")}
FALLBACK_TOKEN_RE = re.compile(r"\S+")
DEFAULT_THINK_LENGTH_ANCHOR_KEYS = ("gt_cot", "gt_response", "gt_solution", "solution")

DEFAULT_FORMAT_REWARD_BONUS = 0.05
DEFAULT_STRICT_FORMAT_GATE = True
DEFAULT_THINK_PENALTY_HARD_TOKENS = 64
DEFAULT_THINK_PENALTY_TARGET_TOKENS = 256
DEFAULT_THINK_PENALTY_MAX_VALUE = -0.5
DEFAULT_INVALID_FORMAT_PENALTY = -0.6
DEFAULT_THINK_PENALTY_DYNAMIC = True
DEFAULT_THINK_PENALTY_TARGET_MIN_TOKENS = 96
DEFAULT_THINK_PENALTY_TARGET_MAX_TOKENS = 512
DEFAULT_THINK_PENALTY_TARGET_RATIO = 0.5
DEFAULT_THINK_PENALTY_HARD_RATIO = 0.5

TOKENIZER_CACHE = {}


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def _get_env_int_with_legacy(name: str, default: int, legacy_name: str | None = None, legacy_divisor: int = 1) -> int:
    value = os.environ.get(name)
    if value is not None and value != "":
        return int(value)

    if legacy_name is None:
        return default

    legacy_value = os.environ.get(legacy_name)
    if legacy_value is None or legacy_value == "":
        return default
    return max(1, int(legacy_value) // legacy_divisor)


def _get_env_float_with_legacy(name: str, default: float, legacy_name: str | None = None) -> float:
    value = os.environ.get(name)
    if value is not None and value != "":
        return float(value)

    if legacy_name is None:
        return default

    legacy_value = os.environ.get(legacy_name)
    if legacy_value is None or legacy_value == "":
        return default
    return float(legacy_value)


def _strip_trailing_special_tokens(text: str, preserve_tags: set[str] | None = None) -> str:
    if not isinstance(text, str):
        return text

    preserve_tags = preserve_tags or set()
    end = len(text)
    while True:
        while end > 0 and text[end - 1].isspace():
            end -= 1

        if end < 4 or text[end - 2 : end] != "|>":
            break

        start = text.rfind("<|", 0, end - 2)
        if start < 0:
            break

        candidate = text[start:end]
        inner = candidate[2:-2]
        if candidate in preserve_tags or not inner or any(ch in "<>|" for ch in inner):
            break

        end = start

    return text[:end].strip()


def _parse_strict_response(response: str) -> tuple[str, str, str] | None:
    if not isinstance(response, str):
        return None

    normalized_response = _strip_trailing_special_tokens(response, preserve_tags=STRICT_FORMAT_SPECIAL_TOKEN_TAGS)
    if not normalized_response:
        return None

    think_open, think_close, answer_open, answer_close = STRICT_FORMAT_TAGS
    think_open_pos = normalized_response.find(think_open)
    if think_open_pos < 0 or normalized_response.find(think_open, think_open_pos + 1) >= 0:
        return None

    think_close_pos = normalized_response.find(think_close, think_open_pos + len(think_open))
    if think_close_pos < 0 or normalized_response.find(think_close, think_close_pos + 1) >= 0:
        return None

    answer_open_pos = normalized_response.find(answer_open, think_close_pos + len(think_close))
    if answer_open_pos < 0 or normalized_response.find(answer_open, answer_open_pos + 1) >= 0:
        return None

    answer_close_pos = normalized_response.find(answer_close, answer_open_pos + len(answer_open))
    if answer_close_pos < 0 or normalized_response.find(answer_close, answer_close_pos + 1) >= 0:
        return None

    if normalized_response[:think_open_pos].strip():
        return None
    if normalized_response[think_close_pos + len(think_close) : answer_open_pos].strip():
        return None
    if normalized_response[answer_close_pos + len(answer_close) :].strip():
        return None

    think_content = normalized_response[think_open_pos + len(think_open) : think_close_pos]
    answer_content = normalized_response[answer_open_pos + len(answer_open) : answer_close_pos]
    return normalized_response, think_content, answer_content


def _extract_strict_response_contents(response: str) -> tuple[str, str] | None:
    parsed_response = _parse_strict_response(response)
    if parsed_response is None:
        return None
    _, think_content, answer_content = parsed_response
    return think_content, answer_content


def _extract_tagged_text(text: str, open_tag: str, close_tag: str) -> str | None:
    if not isinstance(text, str):
        return None

    preserve_tags = {tag for tag in (open_tag, close_tag) if tag.startswith("<|") and tag.endswith("|>")}
    normalized_text = _strip_trailing_special_tokens(text, preserve_tags=preserve_tags)
    open_index = normalized_text.find(open_tag)
    close_index = normalized_text.find(close_tag, open_index + len(open_tag))
    if (
        open_index < 0
        or close_index < 0
        or normalized_text.find(open_tag, open_index + 1) >= 0
        or normalized_text.find(close_tag, close_index + 1) >= 0
    ):
        return None
    if open_index < 0 or close_index < open_index + len(open_tag):
        return None
    return normalized_text[open_index + len(open_tag) : close_index]


def _get_reward_tokenizer(args):
    tokenizer_path = getattr(args, "tokenizer_model", None) or getattr(args, "hf_checkpoint", None)
    if tokenizer_path is None or tokenizer_path == "":
        return None

    if tokenizer_path not in TOKENIZER_CACHE:
        TOKENIZER_CACHE[tokenizer_path] = load_tokenizer(tokenizer_path, trust_remote_code=True)
    return TOKENIZER_CACHE[tokenizer_path]


def _count_text_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(FALLBACK_TOKEN_RE.findall(text))
    return len(tokenizer.encode(text, add_special_tokens=False))


def _extract_anchor_think_text(anchor_text: str) -> str:
    think_text = _extract_tagged_text(anchor_text, STRICT_FORMAT_TAGS[0], STRICT_FORMAT_TAGS[1])
    if think_text is not None:
        return think_text.strip()
    return anchor_text.strip()


def _get_response_think_token_span(tokenizer, think_content: str | None) -> tuple[int, int]:
    if tokenizer is None or think_content is None:
        return 0, 0

    think_open_token_count = len(tokenizer.encode(STRICT_FORMAT_TAGS[0], add_special_tokens=False))
    think_token_count = len(tokenizer.encode(think_content, add_special_tokens=False))
    return think_open_token_count, think_open_token_count + think_token_count


def _get_anchor_think_length_tokens(metadata: dict | None, tokenizer) -> int:
    if not isinstance(metadata, dict):
        return 0

    for key in DEFAULT_THINK_LENGTH_ANCHOR_KEYS:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        anchor_think_text = _extract_anchor_think_text(value)
        if anchor_think_text:
            return _count_text_tokens(anchor_think_text, tokenizer)

    return 0


def _resolve_dynamic_think_thresholds(metadata: dict | None, tokenizer) -> tuple[int, int, int]:
    fallback_hard = _get_env_int_with_legacy(
        "GAR_THINK_PENALTY_HARD_TOKENS",
        DEFAULT_THINK_PENALTY_HARD_TOKENS,
        legacy_name="GAR_THINK_PENALTY_SHORT_CHARS",
        legacy_divisor=4,
    )
    fallback_target = _get_env_int_with_legacy(
        "GAR_THINK_PENALTY_TARGET_TOKENS",
        DEFAULT_THINK_PENALTY_TARGET_TOKENS,
        legacy_name="GAR_THINK_PENALTY_MEDIUM_CHARS",
        legacy_divisor=4,
    )
    if not _get_env_bool("GAR_THINK_PENALTY_DYNAMIC", DEFAULT_THINK_PENALTY_DYNAMIC):
        return fallback_hard, fallback_target, 0

    anchor_think_length_tokens = _get_anchor_think_length_tokens(metadata, tokenizer)
    if anchor_think_length_tokens <= 0:
        return fallback_hard, fallback_target, 0

    target_min = _get_env_int_with_legacy(
        "GAR_THINK_PENALTY_TARGET_MIN_TOKENS",
        DEFAULT_THINK_PENALTY_TARGET_MIN_TOKENS,
    )
    target_max = _get_env_int_with_legacy(
        "GAR_THINK_PENALTY_TARGET_MAX_TOKENS",
        DEFAULT_THINK_PENALTY_TARGET_MAX_TOKENS,
    )
    target_ratio = _get_env_float("GAR_THINK_PENALTY_TARGET_RATIO", DEFAULT_THINK_PENALTY_TARGET_RATIO)
    hard_ratio = _get_env_float("GAR_THINK_PENALTY_HARD_RATIO", DEFAULT_THINK_PENALTY_HARD_RATIO)

    assert target_min > 0, f"Invalid GAR_THINK_PENALTY_TARGET_MIN_TOKENS={target_min}."
    assert target_max >= target_min, (
        f"GAR_THINK_PENALTY_TARGET_MAX_TOKENS={target_max} must be >= "
        f"GAR_THINK_PENALTY_TARGET_MIN_TOKENS={target_min}."
    )
    assert 0.0 < target_ratio <= 1.0, f"Invalid GAR_THINK_PENALTY_TARGET_RATIO={target_ratio}."
    assert 0.0 < hard_ratio <= 1.0, f"Invalid GAR_THINK_PENALTY_HARD_RATIO={hard_ratio}."

    target_threshold = int(round(anchor_think_length_tokens * target_ratio))
    target_threshold = min(max(target_threshold, target_min), target_max)
    hard_threshold = int(round(target_threshold * hard_ratio))
    hard_threshold = min(target_threshold, max(1, hard_threshold))
    return hard_threshold, target_threshold, anchor_think_length_tokens


def get_think_length_penalty_info(
    response: str,
    metadata: dict | None = None,
    tokenizer=None,
    parsed_contents: tuple[str, str] | None = None,
) -> tuple[float, int, str]:
    contents = parsed_contents if parsed_contents is not None else _extract_strict_response_contents(response)
    if contents is None:
        return 0.0, 0, "none"

    think_content, _ = contents
    think_length_tokens = _count_text_tokens(think_content, tokenizer)
    hard_threshold, target_threshold, _ = _resolve_dynamic_think_thresholds(metadata, tokenizer)
    max_penalty = _get_env_float_with_legacy(
        "GAR_THINK_PENALTY_MAX_VALUE",
        DEFAULT_THINK_PENALTY_MAX_VALUE,
        legacy_name="GAR_THINK_PENALTY_SHORT_VALUE",
    )

    assert hard_threshold > 0, f"Invalid GAR_THINK_PENALTY_HARD_TOKENS={hard_threshold}."
    assert target_threshold >= hard_threshold, (
        f"GAR_THINK_PENALTY_TARGET_TOKENS={target_threshold} must be >= "
        f"GAR_THINK_PENALTY_HARD_TOKENS={hard_threshold}."
    )
    assert max_penalty <= 0.0, f"Invalid GAR_THINK_PENALTY_MAX_VALUE={max_penalty}."

    if think_length_tokens <= hard_threshold:
        return max_penalty, think_length_tokens, "short"
    if think_length_tokens >= target_threshold:
        return 0.0, think_length_tokens, "none"

    penalty_progress = (target_threshold - think_length_tokens) / (target_threshold - hard_threshold)
    penalty = max_penalty * penalty_progress
    return penalty, think_length_tokens, "medium"


def is_strict_format_valid(response: str) -> bool:
    contents = _extract_strict_response_contents(response)
    if contents is None:
        return False

    think_content, answer_content = contents
    think_content = think_content.strip()
    answer_content = answer_content.strip()
    return bool(think_content) and bool(answer_content)


def _normalize_rewards(args, rewards: list[float]) -> list[float]:
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization
    ):
        return rewards

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)
    if rewards_tensor.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
        rewards_tensor = rewards_tensor.reshape(-1, args.n_samples_per_prompt)
    else:
        rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])

    rewards_tensor = rewards_tensor - rewards_tensor.mean(dim=-1, keepdim=True)

    if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
        rewards_tensor = rewards_tensor / (rewards_tensor.std(dim=-1, keepdim=True) + 1e-6)

    return rewards_tensor.flatten().tolist()


def apply_strict_format_reward(args, samples, **kwargs):
    format_reward_bonus = _get_env_float("GAR_FORMAT_REWARD_BONUS", DEFAULT_FORMAT_REWARD_BONUS)
    strict_format_gate = _get_env_bool("GAR_STRICT_FORMAT_GATE", DEFAULT_STRICT_FORMAT_GATE)
    invalid_format_penalty = _get_env_float("GAR_INVALID_FORMAT_PENALTY", DEFAULT_INVALID_FORMAT_PENALTY)
    tokenizer = None

    raw_rewards = []
    shaped_rewards = []

    for sample in samples:
        original_reward = float(sample.get_reward_value(args))
        contents = _extract_strict_response_contents(sample.response)
        format_valid = contents is not None
        current_invalid_format_penalty = invalid_format_penalty if not format_valid else 0.0
        think_length_penalty = 0.0
        think_length_tokens = 0
        think_length_penalty_level = "none"
        think_hard_tokens = 0
        think_target_tokens = 0
        anchor_think_length_tokens = 0
        think_token_start = 0
        think_token_end = 0

        if format_valid:
            if tokenizer is None:
                tokenizer = _get_reward_tokenizer(args)
            think_length_penalty, think_length_tokens, think_length_penalty_level = get_think_length_penalty_info(
                sample.response,
                metadata=sample.metadata,
                tokenizer=tokenizer,
                parsed_contents=contents,
            )
            think_hard_tokens, think_target_tokens, anchor_think_length_tokens = _resolve_dynamic_think_thresholds(
                sample.metadata, tokenizer
            )
            think_token_start, think_token_end = _get_response_think_token_span(tokenizer, contents[0])

        gated_reward = original_reward if (format_valid or not strict_format_gate) else 0.0
        shaped_reward = (
            gated_reward
            + (format_reward_bonus if format_valid else 0.0)
            + think_length_penalty
            + current_invalid_format_penalty
        )

        sample.metadata["base_raw_reward"] = original_reward
        sample.metadata["format_valid"] = int(format_valid)
        sample.metadata["raw_reward"] = gated_reward
        sample.metadata["format_reward_bonus"] = format_reward_bonus if format_valid else 0.0
        sample.metadata["invalid_format_penalty"] = current_invalid_format_penalty
        sample.metadata["think_length_penalty"] = think_length_penalty
        sample.metadata["think_length_tokens"] = think_length_tokens
        sample.metadata["think_length_penalty_level"] = think_length_penalty_level
        sample.metadata["think_hard_tokens"] = think_hard_tokens
        sample.metadata["think_target_tokens"] = think_target_tokens
        sample.metadata["anchor_think_length_tokens"] = anchor_think_length_tokens
        sample.metadata["think_token_start"] = think_token_start
        sample.metadata["think_token_end"] = think_token_end

        raw_rewards.append(gated_reward)
        shaped_rewards.append(shaped_reward)

    return raw_rewards, _normalize_rewards(args, shaped_rewards)
