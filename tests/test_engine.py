from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from tinyorca.core.request import Request, SamplingConfig


def _build_request(
    request_id: str,
    prompt_ids: tuple[int, ...],
    max_new_tokens: int = 4,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_ids=prompt_ids,
        sampling=SamplingConfig(max_new_tokens=max_new_tokens, eos_token_id=None),
    )


def _hf_prefill_last_logits(
    model,
    prompt_ids: tuple[int, ...],
) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(
            input_ids=torch.tensor([list(prompt_ids)]),
            use_cache=True,
            return_dict=True,
        )
    return outputs.logits[:, -1, :]


def _hf_decode_last_logits(
    model,
    prompt_ids: tuple[int, ...],
) -> tuple[int, torch.Tensor]:
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        prefill_outputs = model(
            input_ids=torch.tensor([list(prompt_ids)]),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        first_token_id = int(torch.argmax(prefill_outputs.logits[0, -1], dim=-1).item())
        decode_outputs = model(
            input_ids=torch.tensor([[first_token_id]]),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    return first_token_id, decode_outputs.logits[:, -1, :]


def test_run_iter_matches_hf_on_prefill(engine_factory) -> None:
    engine = engine_factory()
    request = _build_request("r1", (10, 11, 12))

    hf_logits = _hf_prefill_last_logits(engine.hf_model, request.prompt_ids)
    expected_token_id = int(torch.argmax(hf_logits[0], dim=-1).item())
    token_events = engine.run_iter([request])

    assert [event.token_id for event in token_events] == [expected_token_id]
    assert request.output_ids == [expected_token_id]


def test_run_iter_matches_hf_on_mixed_batch(engine_factory) -> None:
    engine = engine_factory()
    running_request = _build_request("r1", (10, 11, 12))
    waiting_request = _build_request("r2", (20, 21))

    engine.run_iter([running_request])
    token_events = engine.run_iter([running_request, waiting_request])
    token_ids_by_request = {
        event.request.request_id: event.token_id for event in token_events
    }

    _first_token_id, hf_decode_logits = _hf_decode_last_logits(
        engine.hf_model, running_request.prompt_ids
    )
    hf_prefill_logits = _hf_prefill_last_logits(
        engine.hf_model, waiting_request.prompt_ids
    )

    assert token_ids_by_request[running_request.request_id] == int(
        torch.argmax(hf_decode_logits[0], dim=-1).item()
    )
    assert token_ids_by_request[waiting_request.request_id] == int(
        torch.argmax(hf_prefill_logits[0], dim=-1).item()
    )


def test_selective_model_matches_hf_next_tokens_on_mixed_batch(engine_factory) -> None:
    engine = engine_factory()
    running_request = _build_request("r1", (10, 11, 12))
    waiting_request = _build_request("r2", (20, 21))

    engine.run_iter([running_request])
    flat_batch = engine.build_flat_batch([running_request, waiting_request])

    with torch.inference_mode():
        output_hidden_states = engine.model(
            hidden_states=flat_batch.hidden_states,
            spans=flat_batch.spans,
            position_ids=flat_batch.position_ids,
            cache_position=flat_batch.cache_position,
            request_caches=engine.request_caches,
        )

    last_hidden_states = torch.stack(
        [output_hidden_states[span.end - 1] for span in flat_batch.spans]
    )
    selective_next_token_ids = torch.argmax(
        engine.hf_model.lm_head(last_hidden_states),
        dim=-1,
    ).tolist()

    _first_token_id, hf_decode_logits = _hf_decode_last_logits(
        engine.hf_model, running_request.prompt_ids
    )
    hf_prefill_logits = _hf_prefill_last_logits(
        engine.hf_model, waiting_request.prompt_ids
    )
    expected_next_token_ids = [
        int(torch.argmax(hf_decode_logits[0], dim=-1).item()),
        int(torch.argmax(hf_prefill_logits[0], dim=-1).item()),
    ]

    assert selective_next_token_ids == expected_next_token_ids
