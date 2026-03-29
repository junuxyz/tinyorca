from __future__ import annotations

import pytest

from tinyorca.core.request import RequestState, SamplingConfig
from tinyorca.core.scheduler import RequestPool
from tinyorca.core.serve import Endpoint


def test_endpoint_submit_enqueues_request(tokenizer) -> None:
    request_pool = RequestPool()
    endpoint = Endpoint(tokenizer=tokenizer, request_pool=request_pool)

    request = endpoint.submit(
        prompt_text="ab",
        sampling=SamplingConfig(max_new_tokens=2, eos_token_id=tokenizer.eos_token_id),
    )

    assert request.request_id == "req-0"
    assert request.prompt_ids == tuple(
        tokenizer.encode("ab", add_special_tokens=False, verbose=False)
    )
    assert request.metrics.submitted_at is not None
    assert request_pool.arrival_ordered_requests() == [request]


def test_generate_streams_interleaved_tokens(serve_factory) -> None:
    serve = serve_factory(estimated_n_slots=32, max_batch_size=2, max_new_tokens=2)

    token_events = list(
        serve.generate(
            ["hi", "orca"],
            sampling=SamplingConfig(max_new_tokens=2, eos_token_id=None),
        )
    )
    requests = {event.request.request_id: event.request for event in token_events}

    request_ids = [event.request.request_id for event in token_events]
    assert request_ids == ["req-0", "req-1", "req-0", "req-1"]
    assert len(token_events) == 4
    assert set(requests) == {"req-0", "req-1"}
    assert all(
        request.sampling.eos_token_id == serve.sampling.eos_token_id
        for request in requests.values()
    )
    assert all(request.state is RequestState.FINISHED for request in requests.values())
    assert all(len(request.output_ids) == 2 for request in requests.values())
    assert serve.request_pool.has_requests() is False


def test_generate_rejects_too_few_sampling_configs(serve_factory) -> None:
    serve = serve_factory()

    with pytest.raises(ValueError, match="sampling must match the number of prompts"):
        list(
            serve.generate(
                ["a", "b"],
                sampling=[SamplingConfig(max_new_tokens=1, eos_token_id=0)],
            )
        )


def test_generate_rejects_too_many_sampling_configs(serve_factory) -> None:
    serve = serve_factory()

    with pytest.raises(ValueError, match="sampling must match the number of prompts"):
        list(
            serve.generate(
                ["a"],
                sampling=[
                    SamplingConfig(max_new_tokens=1, eos_token_id=0),
                    SamplingConfig(max_new_tokens=2, eos_token_id=0),
                ],
            )
        )


def test_generate_rejects_request_exceeding_total_slot_capacity(
    serve_factory,
) -> None:
    serve = serve_factory(
        estimated_n_slots=4,
        max_new_tokens=2,
    )

    with pytest.raises(ValueError, match="request exceeds total n_slots capacity"):
        list(
            serve.generate(
                ["hello"],
                sampling=SamplingConfig(max_new_tokens=2, eos_token_id=None),
            )
        )
