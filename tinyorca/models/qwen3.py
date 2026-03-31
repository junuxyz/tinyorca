from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, Qwen3ForCausalLM, apply_rotary_pos_emb, eager_attention_forward


@dataclass(slots=True)
class RequestSpan:
    """One request slice inside the flattened [sum_S, Hidden] tensor."""

    request_id: str
    start: int
    end: int

# Split
def split_hidden_states(hidden_states: torch.Tensor, spans: list[RequestSpan]) -> list[torch.Tensor]:
    return [hidden_states[span.start:span.end].unsqueeze(0) for span in spans]


def prepare_attention_inputs(
    attention_module: nn.Module,
    hidden_states: torch.Tensor,
    spans: list[RequestSpan],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Add a singleton batch dimension so the projection/reshape path stays
    # aligned with Hugging Face attention's convention([B, S, Hidden]).
    hidden_states_batched = hidden_states.unsqueeze(0)
    input_shape = hidden_states_batched.shape[:-1]
    # shape: [B=1, sum_S, Head, HiddenHead]
    hidden_shape = (*input_shape, -1, attention_module.head_dim)

    # Qwen3 normalizes Q and K but not V.
    query_states = attention_module.q_norm(
        attention_module.q_proj(hidden_states_batched).view(hidden_shape)
    ).transpose(1, 2)
    key_states = attention_module.k_norm(
        attention_module.k_proj(hidden_states_batched).view(hidden_shape)
    ).transpose(1, 2)
    value_states = attention_module.v_proj(hidden_states_batched).view(hidden_shape).transpose(1, 2)

    request_qkv_slices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for span in spans:
        request_qkv_slices.append(
            (
                query_states[:, :, span.start : span.end, :].contiguous(),
                key_states[:, :, span.start : span.end, :].contiguous(),
                value_states[:, :, span.start : span.end, :].contiguous(),
            )
        )
    return request_qkv_slices


def run_request_attention(
    attention_module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor | None,
    request_cache: DynamicCache,
) -> torch.Tensor:
    input_shape = (query_states.shape[0], query_states.shape[2])  # [B, S]
    cos, sin = position_embeddings
    # RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states, value_states = request_cache.update(key_states, value_states, attention_module.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attention_module.config._attn_implementation,
        # eager attention as default
        eager_attention_forward,
    )
    # attention operation
    attn_output, _attn_weights = attention_interface(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attention_module.training else attention_module.attention_dropout,
        scaling=attention_module.scaling,
        sliding_window=attention_module.sliding_window,
        cache_position=cache_position,
    )
    # [B=1, Head, S, HiddenHead] -> [B=1, S, Head * HiddenHead]
    # squeeze(0) -> [S, Head * HiddenHead]
    return attn_output.reshape(*input_shape, -1).contiguous().squeeze(0)

# Merge
def merge_request_outputs(
    spans: list[RequestSpan],
    request_outputs: list[torch.Tensor],
    n_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    merged_output = torch.empty((n_tokens, hidden_size), dtype=dtype, device=device)
    for span, request_output in zip(spans, request_outputs, strict=True):
        merged_output[span.start : span.end] = request_output.squeeze(0)
    return merged_output


class Qwen3SelectiveModel(nn.Module):
    """Qwen3 forward path that owns execution block-by-block."""

    def __init__(self, model: Qwen3ForCausalLM):
        super().__init__()
        self.model = model
        self.layers = model.model.layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        spans: list[RequestSpan],
        position_ids: list[torch.Tensor],
        cache_position: list[torch.Tensor],
        request_caches: dict[str, DynamicCache],
    ) -> torch.Tensor:
        request_hidden_states = split_hidden_states(hidden_states, spans)
        position_embeddings: list[tuple[torch.Tensor, torch.Tensor]] = []
        attention_masks: list[torch.Tensor | None] = []

        # build each request's RoPE inputs and causal mask once for this step.
        for req_hidden, span, req_position_ids, req_cache_position in zip(
            request_hidden_states, spans, position_ids, cache_position, strict=True
        ):
            req_id = span.request_id
            req_position_ids = req_position_ids.unsqueeze(0)
            position_embeddings.append(self.model.model.rotary_emb(req_hidden, req_position_ids))
            attention_masks.append(
                create_causal_mask(
                    config=self.model.config,
                    inputs_embeds=req_hidden,
                    attention_mask=None,
                    cache_position=req_cache_position,
                    past_key_values=request_caches[req_id],
                    position_ids=req_position_ids,
                )
            )

        # for each Transformer Layer
        for layer in self.layers:
            residual = hidden_states
            # RMSNorm 1
            hidden_states = layer.input_layernorm(hidden_states)

            # batched Q, K, V projection for the flattened token stream.
            request_qkv_slices = prepare_attention_inputs(layer.self_attn, hidden_states, spans)
            request_outputs: list[torch.Tensor] = []

            # GQA (Grouped-Query Attention)
            # Only RoPE, cache update, mask handling, and the attention kernel
            # stay request-local.
            for (req_query_states, req_key_states, req_value_states), span, req_position_embeddings, req_cache_position, attention_mask in zip(
                request_qkv_slices,
                spans,
                position_embeddings,
                cache_position,
                attention_masks,
                strict=True,
            ):
                req_id = span.request_id
                attn_out = run_request_attention(
                    layer.self_attn,
                    query_states=req_query_states,
                    key_states=req_key_states,
                    value_states=req_value_states,
                    position_embeddings=req_position_embeddings,
                    cache_position=req_cache_position,
                    attention_mask=attention_mask,
                    request_cache=request_caches[req_id],
                )
                request_outputs.append(attn_out)

            # Merge per-request attention outputs back into the flat tensor.
            attn_output = merge_request_outputs(
                spans=spans,
                request_outputs=request_outputs,
                n_tokens=hidden_states.shape[0],
                hidden_size=layer.self_attn.o_proj.in_features,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attn_output = layer.self_attn.o_proj(attn_output)

            # residual connection
            hidden_states = residual + attn_output
            residual = hidden_states
            # RMSNorm 2
            hidden_states = layer.post_attention_layernorm(hidden_states)
            # MLP
            hidden_states = layer.mlp(hidden_states)
            # residual connection
            hidden_states = residual + hidden_states
        # final RMSNorm
        return self.model.model.norm(hidden_states)
