from __future__ import annotations

from tinyorca.core.request import Request, SamplingConfig
from tinyorca.core.scheduler import OrcaScheduler, RequestPool


def _build_request(
    request_id: str,
    prompt_ids: tuple[int, ...],
    max_new_tokens: int,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_ids=prompt_ids,
        sampling=SamplingConfig(max_new_tokens=max_new_tokens, eos_token_id=None),
    )


def test_select_preserves_iteration_level_fcfs(engine_factory) -> None:
    engine = engine_factory(n_slots=32)
    earlier_request = _build_request("r1", (10, 11, 12), max_new_tokens=2)
    later_request = _build_request("r2", (20, 21), max_new_tokens=2)

    engine.run_iter([earlier_request])
    earlier_request.increment()

    request_pool = RequestPool()
    request_pool.push(earlier_request)
    request_pool.push(later_request)

    scheduler = OrcaScheduler(engine, request_pool, max_batch_size=2)

    selected = scheduler.select()

    assert [request.request_id for request in selected] == ["r1", "r2"]


def test_schedule_admits_later_request_after_slots_free_up(engine_factory) -> None:
    engine = engine_factory(n_slots=7, max_new_tokens=2)
    request_pool = RequestPool()

    for request in (
        _build_request("r0", (1, 2), max_new_tokens=2),
        _build_request("r1", (3,), max_new_tokens=2),
        _build_request("r2", (4,), max_new_tokens=1),
    ):
        request_pool.push(request)

    scheduler = OrcaScheduler(engine, request_pool, max_batch_size=2)

    token_events = list(scheduler.schedule())
    request_ids = [event.request.request_id for event in token_events]
    assert request_ids == ["r0", "r1", "r0", "r1", "r2"]
