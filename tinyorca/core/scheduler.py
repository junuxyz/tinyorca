from __future__ import annotations

from collections.abc import Iterator

from tinyorca.core.engine import OrcaEngine
from tinyorca.core.request import Request, RequestState, RequestToken


class RequestPool:
    """
    Registry of queued and running requests in arrival order.
    """

    def __init__(self):
        self._requests: list[Request] = []

    def push(self, request: Request) -> None:
        self._requests.append(request)

    def arrival_ordered_requests(self) -> list[Request]:
        # ORCA's iteration-level FCFS policy preserves request arrival order
        # across iterations instead of preferring decode requests globally.
        return list(self._requests)

    def remove(self, request: Request) -> None:
        self._requests.remove(request)

    def has_requests(self) -> bool:
        return bool(self._requests)


class OrcaScheduler:
    """
    Implementation of Algorithm 1 in the ORCA paper.
    Owns request selection and slot-based admission control.
    """

    def __init__(
        self,
        engine: OrcaEngine,
        request_pool: RequestPool,
        max_batch_size: int,
    ):
        self.engine = engine
        self.request_pool = request_pool
        self.max_batch_size = max_batch_size
        n_slots = engine.config.n_slots
        if n_slots is None:
            max_new_tokens = engine.config.max_new_tokens
            n_slots = engine.estimate_n_slots(max_batch_size, max_new_tokens)
        self.n_slots = n_slots
        if self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        self.n_rsrv = 0

    @staticmethod
    def request_slots(request: Request) -> int:
        return len(request.prompt_ids) + request.sampling.max_new_tokens

    def select(self) -> list[Request]:
        """
        Implementation of `Select(pool, n_rsrv)` in Algorithm 1.
        """
        batch: list[Request] = []
        for request in self.request_pool.arrival_ordered_requests():
            if len(batch) == self.max_batch_size:
                break
            if request.state is RequestState.INITIATION:
                request_slots = self.request_slots(request)
                if request_slots > self.n_slots:
                    raise ValueError(
                        "request exceeds total n_slots capacity "
                        f"(request_id={request.request_id}, "
                        f"required_slots={request_slots}, n_slots={self.n_slots})"
                    )
                # reserve KV only once, when a request is first admitted.
                new_n_rsrv = self.n_rsrv + request_slots
                if new_n_rsrv > self.n_slots:
                    break
                self.n_rsrv = new_n_rsrv
            batch.append(request)
        return batch

    def schedule(self) -> Iterator[RequestToken]:
        """
        Implementation of scheduler loop in Algorithm 1.
        """
        while self.request_pool.has_requests():
            batch = self.select()
            if not batch:
                break
            token_events = self.engine.run_iter(batch)
            for token_event in token_events:
                request = token_event.request
                if request.state is RequestState.FINISHED:
                    self.request_pool.remove(request)
                    self.n_rsrv -= self.request_slots(request)
                else:
                    request.increment()
                yield token_event
