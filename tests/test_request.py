from __future__ import annotations

from tinyorca.core.request import FinishReason, Request, RequestState, SamplingConfig


def test_record_token_tracks_first_token_and_finishes_at_max_new_tokens() -> None:
    request = Request(
        request_id="r1",
        prompt_ids=(1, 2),
        sampling=SamplingConfig(max_new_tokens=2, eos_token_id=None),
    )

    request.mark_submitted(now=1.0)
    request.record_token(10, now=2.0)
    request.record_token(11, now=3.0)

    assert request.output_ids == [10, 11]
    assert request.state is RequestState.FINISHED
    assert request.finish_reason is FinishReason.MAX_NEW_TOKENS
    assert request.metrics.submitted_at == 1.0
    assert request.metrics.first_token_at == 2.0
    assert request.metrics.finished_at == 3.0


def test_record_token_finishes_early_on_eos() -> None:
    request = Request(
        request_id="r2",
        prompt_ids=(1,),
        sampling=SamplingConfig(max_new_tokens=4, eos_token_id=99),
    )

    request.record_token(99, now=5.0)

    assert request.output_ids == [99]
    assert request.state is RequestState.FINISHED
    assert request.finish_reason is FinishReason.EOS
    assert request.metrics.first_token_at == 5.0
    assert request.metrics.finished_at == 5.0
