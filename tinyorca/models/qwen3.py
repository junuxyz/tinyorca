from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
)


@dataclass(slots=True)
class RequestSpan:
    """One request slice inside the flattened [sum_S, Hidden] tensor."""

    request_id: str
    start: int
    end: int


class Qwen3SelectiveModel(nn.Module):
    """
    Qwen3 forward path that owns execution block-by-block.
    """

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
        position_embeddings: list[tuple[torch.Tensor, torch.Tensor]] = []
        attention_masks: list[torch.Tensor | None] = []

        for span, req_position_ids, req_cache_position in zip(
            spans,
            position_ids,
            cache_position,
            strict=True,
        ):
            req_id = span.request_id
            req_hidden = hidden_states[span.start : span.end].unsqueeze(0)
            req_position_ids = req_position_ids.unsqueeze(0)
            position_embeddings.append(
                self.model.model.rotary_emb(
                    req_hidden,
                    req_position_ids,
                )
            )
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

        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            # Selective batching keeps flat token-wise ops batched, but runs
            # attention request by request so each request can use its own KV.
            attn_output = torch.empty_like(hidden_states)
            for span, req_position_embeddings, req_cache_position, attention_mask in zip(
                spans,
                position_embeddings,
                cache_position,
                attention_masks,
                strict=True,
            ):
                req_id = span.request_id
                req_hidden = hidden_states[span.start : span.end].unsqueeze(0)
                attn_out, _attn_weights = layer.self_attn(
                    hidden_states=req_hidden,
                    position_embeddings=req_position_embeddings,
                    cache_position=req_cache_position,
                    attention_mask=attention_mask,
                    past_key_values=request_caches[req_id],
                )
                attn_output[span.start : span.end] = attn_out.squeeze(0)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
        return self.model.model.norm(hidden_states)
