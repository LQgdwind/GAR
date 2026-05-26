from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

DEFAULT_DATA_PAD_SIZE_MULTIPLIER = 128
THINK_OPEN_TAG = "<|think|>"
THINK_CLOSE_TAG = "<|/think|>"
PRESERVED_TRAILING_SPECIAL_TAGS = {THINK_OPEN_TAG, THINK_CLOSE_TAG}


DEFAULT_GROUND_TRUTH_KEYS = (
    "gt_response",
    "gt_cot",
    "gt_solution",
    "solution",
)

DEFAULT_AUX_KEYS = (
    "gar_aux_responses",
    "teacher_roll_responses",
    "teacher_anchors",
)


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


def _get_last_model_chunk(model_chunks: Sequence):
    assert len(model_chunks) == 1, "GAR online reward currently supports a single Megatron model chunk only."
    model_chunk = model_chunks[-1]
    assert hasattr(model_chunk, "module"), "Expected Megatron DDP wrapper with a `.module` attribute."
    return model_chunk


def _get_unwrapped_model_module(model_chunk):
    model_module = model_chunk.module
    while hasattr(model_module, "module"):
        model_module = model_module.module
    return model_module


def _get_output_layer_kwargs(model_module):
    output_weight = None
    if getattr(model_module, "share_embeddings_and_output_weights", False):
        output_weight = model_module.shared_embedding_or_output_weight().detach()
    return {
        "weight": output_weight,
        "runtime_gather_output": None,
    }


def _get_data_pad_size(args) -> int:
    pad_multiplier = int(getattr(args, "data_pad_size_multiplier", DEFAULT_DATA_PAD_SIZE_MULTIPLIER))
    assert pad_multiplier > 0, f"Invalid data_pad_size_multiplier={pad_multiplier}."
    return mpu.get_tensor_model_parallel_world_size() * pad_multiplier


def _compute_single_sample_pad(total_length: int, pad_size: int) -> int:
    assert total_length > 0, f"Invalid total_length={total_length}."
    assert pad_size > 0, f"Invalid pad_size={pad_size}."
    return (pad_size - total_length % pad_size) % pad_size


def _truncate_token_range(token_start: int, token_end: int, max_target_tokens: int) -> tuple[int, int]:
    assert 0 <= token_start < token_end, f"Invalid token range: {token_start}, {token_end}."
    if max_target_tokens <= 0 or token_end - token_start <= max_target_tokens:
        return token_start, token_end
    return token_end - max_target_tokens, token_end


def _compute_response_spans(
    total_length: int,
    response_length: int,
    max_response_tokens: int,
) -> tuple[int, int, int, int]:
    assert total_length > 0, f"Invalid total_length={total_length}."
    assert 0 < response_length < total_length, f"Invalid response_length={response_length} for total_length={total_length}."

    token_start = total_length - response_length
    token_end = total_length
    token_start, token_end = _truncate_token_range(token_start, token_end, max_response_tokens)
    hidden_start = token_start - 1
    hidden_end = token_end - 1
    return hidden_start, hidden_end, token_start, token_end


def _compute_target_spans(
    total_length: int,
    response_length: int,
    target_token_start_in_response: int,
    target_token_end_in_response: int,
    max_target_tokens: int,
) -> tuple[int, int, int, int]:
    assert total_length > 0, f"Invalid total_length={total_length}."
    assert 0 < response_length < total_length, f"Invalid response_length={response_length} for total_length={total_length}."
    assert 0 <= target_token_start_in_response < target_token_end_in_response <= response_length, (
        f"Invalid target token range {target_token_start_in_response}:{target_token_end_in_response} "
        f"for response_length={response_length}."
    )

    prompt_length = total_length - response_length
    token_start = prompt_length + target_token_start_in_response
    token_end = prompt_length + target_token_end_in_response
    token_start, token_end = _truncate_token_range(token_start, token_end, max_target_tokens)
    hidden_start = token_start - 1
    hidden_end = token_end - 1
    return hidden_start, hidden_end, token_start, token_end


def _get_effective_response_length(response_length: int, max_response_tokens: int) -> int:
    assert response_length > 0, f"Invalid response_length={response_length}."
    if max_response_tokens <= 0:
        return response_length
    return min(response_length, max_response_tokens)


def _compute_sequence_shard_bounds(
    padded_length: int,
    tp_rank: int,
    tp_size: int,
) -> tuple[int, int]:
    assert padded_length > 0, f"Invalid padded_length={padded_length}."
    assert tp_size > 0, f"Invalid tp_size={tp_size}."
    assert 0 <= tp_rank < tp_size, f"Invalid tp_rank={tp_rank} for tp_size={tp_size}."
    assert padded_length % tp_size == 0, f"{padded_length} must be divisible by tp_size={tp_size}."

    shard_size = padded_length // tp_size
    shard_start = shard_size * tp_rank
    shard_end = shard_start + shard_size
    return shard_start, shard_end


def _compute_local_response_spans(
    hidden_start: int,
    hidden_end: int,
    shard_start: int,
    shard_end: int,
) -> tuple[int, int, int, int] | None:
    local_hidden_start = max(hidden_start, shard_start)
    local_hidden_end = min(hidden_end, shard_end)
    if local_hidden_start >= local_hidden_end:
        return None

    local_token_start = local_hidden_start + 1
    local_token_end = local_hidden_end + 1
    return (
        local_hidden_start - shard_start,
        local_hidden_end - shard_start,
        local_token_start,
        local_token_end,
    )


def _gather_hidden_states_across_tp(hidden_states: torch.Tensor, padded_length: int) -> torch.Tensor:
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size <= 1 or hidden_states.size(0) == padded_length:
        return hidden_states

    tp_rank = mpu.get_tensor_model_parallel_rank()
    shard_start, shard_end = _compute_sequence_shard_bounds(padded_length, tp_rank, tp_size)
    expected_shard_length = shard_end - shard_start
    assert hidden_states.size(0) == expected_shard_length, f"{hidden_states.size(0)} vs {expected_shard_length}"

    gathered_hidden_states = [torch.empty_like(hidden_states) for _ in range(tp_size)]
    dist.all_gather(gathered_hidden_states, hidden_states.contiguous(), group=mpu.get_tensor_model_parallel_group())
    full_hidden_states = torch.cat(gathered_hidden_states, dim=0)
    assert full_hidden_states.size(0) == padded_length, f"{full_hidden_states.size(0)} vs {padded_length}"
    return full_hidden_states


def _make_single_sample_inputs(args, full_tokens: torch.Tensor):
    total_length = int(full_tokens.size(0))
    pad = _compute_single_sample_pad(total_length, _get_data_pad_size(args))
    padded_tokens = full_tokens
    cu_seqlens_list = [0, total_length]
    if pad != 0:
        padded_tokens = F.pad(padded_tokens, (0, pad), value=0)
        cu_seqlens_list.append(total_length + pad)

    tokens = padded_tokens.unsqueeze(0)
    loss_mask = torch.ones_like(tokens, dtype=torch.int)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=full_tokens.device)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
    return tokens, loss_mask, packed_seq_params, int(padded_tokens.size(0))


def _normalize_anchor_value(value) -> list[str]:
    anchors = []
    if isinstance(value, str):
        if value.strip():
            anchors.append(value)
        return anchors

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                anchors.append(item)
        return anchors

    return anchors


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


def _extract_anchor_supervision_text(anchor_text: str) -> str:
    think_text = _extract_tagged_text(anchor_text, THINK_OPEN_TAG, THINK_CLOSE_TAG)
    if think_text is not None:
        return think_text.strip()
    return _strip_trailing_special_tokens(anchor_text, preserve_tags=PRESERVED_TRAILING_SPECIAL_TAGS)


def _get_response_target_token_range(metadata: dict, response_length: int) -> tuple[int | None, int | None]:
    target_start = int(metadata.get("think_token_start", 0))
    target_end = int(metadata.get("think_token_end", 0))
    if target_start < 0 or target_end <= target_start or target_end > response_length:
        return None, None
    return target_start, target_end


def _deduplicate_anchors(anchors: list[str], max_count: int) -> list[str]:
    seen = set()
    deduplicated = []
    for anchor in anchors:
        normalized = " ".join(anchor.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(anchor)
        if max_count > 0 and len(deduplicated) >= max_count:
            break
    return deduplicated


def _get_primary_anchor_texts(args, metadata: dict) -> list[str]:
    candidate_keys = [args.gar_ground_truth_key]
    candidate_keys.extend([key for key in DEFAULT_GROUND_TRUTH_KEYS if key != args.gar_ground_truth_key])

    anchors = []
    for key in candidate_keys:
        anchors.extend(_normalize_anchor_value(metadata.get(key)))

    return _deduplicate_anchors(anchors, args.gar_max_anchors_per_type)


def _get_aux_anchor_texts(args, metadata: dict) -> list[str]:
    candidate_keys = [args.gar_aux_ground_truth_key]
    candidate_keys.extend([key for key in DEFAULT_AUX_KEYS if key != args.gar_aux_ground_truth_key])

    anchors = []
    for key in candidate_keys:
        anchors.extend(_normalize_anchor_value(metadata.get(key)))

    return _deduplicate_anchors(anchors, args.gar_max_anchors_per_type)


def _slice_response_hidden_and_targets(
    hidden_states: torch.Tensor,
    full_tokens: torch.Tensor,
    padded_length: int,
    response_length: int,
    max_response_tokens: int,
    target_token_start_in_response: int | None = None,
    target_token_end_in_response: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Match the response alignment used by Megatron policy log-prob computation:
    # hidden[start - 1 : end - 1] predicts tokens[start : end].
    assert hidden_states.dim() == 3 and hidden_states.size(1) == 1, f"Unexpected hidden shape: {hidden_states.shape}"

    if target_token_start_in_response is not None and target_token_end_in_response is not None:
        hidden_start, hidden_end, _, _ = _compute_target_spans(
            int(full_tokens.size(0)),
            response_length,
            target_token_start_in_response,
            target_token_end_in_response,
            max_response_tokens,
        )
    else:
        hidden_start, hidden_end, _, _ = _compute_response_spans(
            int(full_tokens.size(0)),
            response_length,
            max_response_tokens,
        )

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    shard_start = 0
    shard_end = padded_length
    if hidden_states.size(0) != padded_length:
        shard_start, shard_end = _compute_sequence_shard_bounds(padded_length, tp_rank, tp_size)
        assert hidden_states.size(0) == shard_end - shard_start, f"{hidden_states.size(0)} vs {shard_end - shard_start}"

    local_spans = _compute_local_response_spans(hidden_start, hidden_end, shard_start, shard_end)
    if local_spans is None:
        empty_hidden = hidden_states[:0]
        empty_tokens = full_tokens[:0]
        return empty_hidden, empty_tokens

    local_hidden_start, local_hidden_end, local_token_start, local_token_end = local_spans
    response_hidden = hidden_states[local_hidden_start:local_hidden_end]
    response_tokens = full_tokens[local_token_start:local_token_end]
    assert response_hidden.size(0) == response_tokens.size(0), f"{response_hidden.size(0)} vs {response_tokens.size(0)}"
    return response_hidden, response_tokens


def _capture_last_hidden(args, model_chunks: Sequence, full_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    model_chunk = _get_last_model_chunk(model_chunks)
    model_module = _get_unwrapped_model_module(model_chunk)
    assert hasattr(model_module, "output_layer"), "GAR online reward requires `output_layer` on the model module."

    captured_hidden = []

    def _hook(_, hook_input):
        hidden_states = hook_input[0]
        captured_hidden.append(hidden_states.detach())

    hook_handle = model_module.output_layer.register_forward_pre_hook(_hook)

    original_modes = [chunk.training for chunk in model_chunks]
    for chunk in model_chunks:
        chunk.eval()

    tokens, loss_mask, packed_seq_params, padded_length = _make_single_sample_inputs(args, full_tokens)
    with torch.no_grad():
        _ = model_chunk(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
        )

    hook_handle.remove()

    for chunk, mode in zip(model_chunks, original_modes, strict=False):
        chunk.train(mode)

    assert len(captured_hidden) == 1, f"Expected one hidden capture, got {len(captured_hidden)}."
    return captured_hidden[0], padded_length


def _compute_gradient_vector(
    args,
    model_chunks: Sequence,
    full_tokens: torch.Tensor,
    response_length: int,
    target_token_start_in_response: int | None = None,
    target_token_end_in_response: int | None = None,
) -> torch.Tensor:
    hidden_states, padded_length = _capture_last_hidden(args, model_chunks, full_tokens)
    model_chunk = _get_last_model_chunk(model_chunks)
    model_module = _get_unwrapped_model_module(model_chunk)
    output_layer = model_module.output_layer
    is_sequence_parallel = bool(getattr(output_layer, "sequence_parallel", False))
    tp_size = mpu.get_tensor_model_parallel_world_size()

    hidden_states_for_loss = hidden_states
    if is_sequence_parallel and tp_size > 1:
        hidden_states_for_loss = _gather_hidden_states_across_tp(hidden_states, padded_length)

    response_hidden, response_tokens = _slice_response_hidden_and_targets(
        hidden_states_for_loss,
        full_tokens,
        padded_length,
        response_length,
        args.gar_max_response_tokens,
        target_token_start_in_response=target_token_start_in_response,
        target_token_end_in_response=target_token_end_in_response,
    )

    hidden_size = int(hidden_states.size(-1))
    response_token_count = int(response_tokens.size(0))
    effective_response_length = response_token_count
    if response_token_count == 0:
        grad_sum = torch.zeros(hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
    else:
        hidden_leaf = response_hidden.detach().requires_grad_(True)
        if is_sequence_parallel and tp_size > 1:
            output_weight = _get_output_layer_kwargs(model_module)["weight"]
            if output_weight is None:
                output_weight = output_layer.weight.detach()
            output_bias = output_layer.bias.detach() if output_layer.bias is not None else None
            local_logits = F.linear(hidden_leaf, output_weight, output_bias)
            per_token_loss = model_module.compute_language_model_loss(
                response_tokens.unsqueeze(0),
                local_logits,
            )
        else:
            output_kwargs = _get_output_layer_kwargs(model_module)
            per_token_loss = model_module.compute_output_layer_and_language_model_loss(
                hidden_leaf,
                labels=response_tokens.unsqueeze(0),
                weight=output_kwargs["weight"],
                sequence_parallel_enabled=is_sequence_parallel,
                column_parallel_linear=output_layer,
                col_linear_kwargs=output_kwargs,
                reduction="none",
            )
        loss = per_token_loss.sum() / effective_response_length
        grad_hidden = torch.autograd.grad(loss, hidden_leaf, retain_graph=False, create_graph=False)[0]
        # GradCAM-style: weight gradient by activation magnitude so the signal
        # encodes both *direction* and *where the model attends*, not just the
        # local loss sensitivity of the output projection.
        grad_act = grad_hidden * hidden_leaf.detach()
        grad_sum = grad_act.sum(dim=(0, 1)).detach()

    if is_sequence_parallel and tp_size > 1:
        dist.all_reduce(grad_sum, group=mpu.get_tensor_model_parallel_group())

    vec = (grad_sum / effective_response_length).detach()
    # L2-normalize so cosine similarity is computed on unit vectors; this also
    # removes gradient magnitude drift across training steps.
    norm = vec.norm(p=2)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def _get_anchor_cache_key(
    args,
    prompt_tokens: torch.Tensor,
    anchor_text: str,
):
    return (tuple(prompt_tokens.tolist()), anchor_text, args.gar_max_response_tokens)


def _build_anchor_gradient(
    args,
    model,
    prompt_tokens: torch.Tensor,
    anchor_text: str,
    tokenizer,
) -> torch.Tensor:
    anchor_tokens = tokenizer.encode(anchor_text, add_special_tokens=False)
    assert anchor_tokens, "GAR anchor tokenization produced an empty sequence."

    full_tokens = torch.cat(
        [
            prompt_tokens,
            torch.tensor(anchor_tokens, dtype=torch.long, device=prompt_tokens.device),
        ],
        dim=0,
    )
    gradient = _compute_gradient_vector(
        args,
        model,
        full_tokens,
        len(anchor_tokens),
    )
    return gradient


def _get_best_cosine(
    args,
    model,
    prompt_tokens: torch.Tensor,
    anchor_texts: list[str],
    pred_gradient: torch.Tensor,
    tokenizer,
    gt_cache: dict,
) -> float | None:
    best_cosine = None
    for anchor_text in anchor_texts:
        effective_anchor_text = _extract_anchor_supervision_text(anchor_text)
        if not effective_anchor_text:
            continue
        cache_key = _get_anchor_cache_key(args, prompt_tokens, effective_anchor_text)
        if cache_key not in gt_cache:
            gt_cache[cache_key] = _build_anchor_gradient(
                args,
                model,
                prompt_tokens,
                effective_anchor_text,
                tokenizer,
            )

        cosine = F.cosine_similarity(pred_gradient.unsqueeze(0), gt_cache[cache_key].unsqueeze(0), dim=-1).item()
        # Keep full [-1, 1] range here; clipping to >= 0 happens after
        # batch-level relative calibration in apply_online_gar.
        best_cosine = cosine if best_cosine is None else max(best_cosine, cosine)

    return best_cosine


def _aggregate_bonus(args, real_best: float | None, aux_best: float | None) -> float:
    if real_best is None:
        assert aux_best is not None
        return aux_best

    if aux_best is None or args.gar_anchor_aggregation == "real_only":
        return real_best

    if args.gar_anchor_aggregation == "aux_max":
        return max(real_best, args.gar_aux_weight * aux_best)

    if args.gar_anchor_aggregation == "aux_residual":
        return real_best + args.gar_aux_weight * max(0.0, aux_best - real_best)

    raise AssertionError(f"Unsupported GAR anchor aggregation: {args.gar_anchor_aggregation}")


def apply_online_gar(args, model, rollout_data, tokenizer) -> None:
    if not mpu.is_pipeline_last_stage():
        return

    print("Starting GAR online reward recomputation")
    assert args.qkv_format == "thd", f"GAR online reward only supports qkv_format='thd', got {args.qkv_format}."
    assert mpu.get_context_parallel_world_size() == 1, "GAR online reward currently supports context parallel size 1."
    assert "metadata" in rollout_data, "GAR online reward requires metadata in rollout_data."

    tokens_list = rollout_data["tokens"]
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    raw_rewards = rollout_data.get("local_raw_reward", rollout_data["raw_reward"])
    metadata_list = rollout_data["metadata"]

    assert len(tokens_list) == len(response_lengths) == len(total_lengths) == len(raw_rewards) == len(metadata_list)

    new_rewards = []
    real_cosine_values = []
    aux_cosine_values = []
    relative_cosine_values = []
    num_aux_only = 0
    think_penalty_values = []
    invalid_format_penalty_values = []
    num_invalid_format = 0
    num_short_think = 0
    num_medium_think = 0
    num_think_span_aligned = 0
    num_response_fallback = 0
    gt_cache = {}

    # Per-sample intermediate results for verifier-passing samples.
    # Each entry: (slot, raw_bonus, think_penalty, invalid_format_penalty, prompt_key)
    passing_sample_results: list[tuple[int, float, float, float, int]] = []

    for tokens, response_length, total_length, raw_reward, metadata in zip(
        tokens_list,
        response_lengths,
        total_lengths,
        raw_rewards,
        metadata_list,
        strict=False,
    ):
        assert isinstance(metadata, dict), f"GAR metadata must be a dict, got {type(metadata)}."
        gate = float(raw_reward)
        format_valid = bool(metadata.get("format_valid", 1))
        invalid_format_penalty = float(metadata.get("invalid_format_penalty", 0.0))
        think_length_penalty = float(metadata.get("think_length_penalty", 0.0))
        think_penalty_values.append(think_length_penalty)
        invalid_format_penalty_values.append(invalid_format_penalty)
        think_penalty_level = metadata.get("think_length_penalty_level", "none")
        if not format_valid:
            num_invalid_format += 1
        if think_penalty_level == "short":
            num_short_think += 1
        elif think_penalty_level == "medium":
            num_medium_think += 1

        if gate <= 0 or not format_valid:
            new_rewards.append(invalid_format_penalty + think_length_penalty)
            continue

        total_length = int(total_length)
        response_length = int(response_length)
        prompt_length = total_length - response_length
        assert prompt_length > 0, f"Invalid prompt_length={prompt_length} for total={total_length}, response={response_length}."

        full_tokens = tokens[:total_length]
        prompt_tokens = full_tokens[:prompt_length]
        target_token_start_in_response, target_token_end_in_response = _get_response_target_token_range(
            metadata, response_length
        )
        if target_token_start_in_response is None or target_token_end_in_response is None:
            num_response_fallback += 1
        else:
            num_think_span_aligned += 1

        primary_anchor_texts = _get_primary_anchor_texts(args, metadata)
        auxiliary_anchor_texts = _get_aux_anchor_texts(args, metadata)
        assert primary_anchor_texts or auxiliary_anchor_texts, (
            "GAR requires at least one primary or auxiliary anchor in metadata. "
            f"Available keys: {sorted(metadata.keys())}"
        )

        pred_gradient = _compute_gradient_vector(
            args,
            model,
            full_tokens,
            response_length,
            target_token_start_in_response=target_token_start_in_response,
            target_token_end_in_response=target_token_end_in_response,
        )

        real_best = _get_best_cosine(
            args,
            model,
            prompt_tokens,
            primary_anchor_texts,
            pred_gradient,
            tokenizer,
            gt_cache,
        )
        aux_best = _get_best_cosine(
            args,
            model,
            prompt_tokens,
            auxiliary_anchor_texts,
            pred_gradient,
            tokenizer,
            gt_cache,
        )

        if real_best is not None:
            real_cosine_values.append(real_best)
        if aux_best is not None:
            aux_cosine_values.append(aux_best)
        if real_best is None and aux_best is not None:
            num_aux_only += 1

        raw_bonus = _aggregate_bonus(args, real_best, aux_best)
        # Placeholder: will be replaced after prompt-group baseline is computed.
        slot = len(new_rewards)
        new_rewards.append(0.0)
        prompt_key = hash(tuple(prompt_tokens.tolist()))
        passing_sample_results.append((slot, raw_bonus, think_length_penalty, invalid_format_penalty, prompt_key))

    # Relative-cosine calibration: subtract the prompt-group mean cosine so
    # that the bonus measures "how much better than sibling rollouts from the
    # same prompt" rather than an absolute cosine value.  This mirrors GRPO's
    # group-level advantage normalisation and prevents easy prompts from
    # dominating the gradient signal.
    # When a prompt group has only one passing sample, relative calibration
    # is degenerate (bonus would always be 0), so fall back to absolute cosine.
    if passing_sample_results:
        # Build per-prompt-group mean.
        from collections import defaultdict
        group_sums: dict[int, float] = defaultdict(float)
        group_counts: dict[int, int] = defaultdict(int)
        for _, raw_bonus, _, _, prompt_key in passing_sample_results:
            group_sums[prompt_key] += raw_bonus
            group_counts[prompt_key] += 1
        group_means = {
            pk: (group_sums[pk] / group_counts[pk]) if group_counts[pk] > 1 else 0.0
            for pk in group_sums
        }
        for slot, raw_bonus, think_length_penalty, invalid_format_penalty, prompt_key in passing_sample_results:
            relative_bonus = raw_bonus - group_means[prompt_key]
            relative_cosine_values.append(relative_bonus)
            new_rewards[slot] = (
                args.gar_base_reward
                + args.gar_beta * max(0.0, relative_bonus)
                + think_length_penalty
                + invalid_format_penalty
            )

    rollout_data["rewards"] = new_rewards

    if real_cosine_values or aux_cosine_values:
        mean_real_cosine = sum(real_cosine_values) / len(real_cosine_values) if real_cosine_values else 0.0
        mean_aux_cosine = sum(aux_cosine_values) / len(aux_cosine_values) if aux_cosine_values else 0.0
        mean_relative_cosine = (
            sum(relative_cosine_values) / len(relative_cosine_values) if relative_cosine_values else 0.0
        )
        mean_reward = sum(new_rewards) / len(new_rewards)
        mean_think_penalty = sum(think_penalty_values) / len(think_penalty_values) if think_penalty_values else 0.0
        mean_invalid_format_penalty = (
            sum(invalid_format_penalty_values) / len(invalid_format_penalty_values)
            if invalid_format_penalty_values
            else 0.0
        )
        print(
            f"Finished GAR online reward recomputation: "
            f"num_real={len(real_cosine_values)}, num_aux={len(aux_cosine_values)}, "
            f"num_aux_only={num_aux_only}, mean_real_cosine={mean_real_cosine:.4f}, "
            f"mean_aux_cosine={mean_aux_cosine:.4f}, mean_relative_cosine={mean_relative_cosine:.4f}, "
            f"mean_reward={mean_reward:.4f}, "
            f"mean_think_penalty={mean_think_penalty:.4f}, "
            f"mean_invalid_format_penalty={mean_invalid_format_penalty:.4f}, "
            f"invalid_format={num_invalid_format}, short_think={num_short_think}, "
            f"medium_think={num_medium_think}, think_span_aligned={num_think_span_aligned}, "
            f"response_fallback={num_response_fallback}"
        )
    else:
        print("Finished GAR online reward recomputation: no verifier-passing samples in this rollout")
