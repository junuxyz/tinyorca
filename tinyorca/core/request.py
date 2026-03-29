from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class RequestState(StrEnum):
    """request lifecycle: INITIATION -> INCREMENT -> FINISHED."""

    INITIATION = "initiation"
    INCREMENT = "increment"
    FINISHED = "finished"


class FinishReason(StrEnum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    max_new_tokens: int = 64
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")


@dataclass(slots=True)
class RequestMetrics:
    submitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None


@dataclass(slots=True)
class Request:
    """Per-request state tracked across admission, prefill, and decode."""

    request_id: str
    prompt_ids: tuple[int, ...]
    sampling: SamplingConfig
    output_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.INITIATION
    finish_reason: FinishReason | None = None
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")

    def mark_submitted(self, now: float | None = None) -> None:
        if self.metrics.submitted_at is None:
            self.metrics.submitted_at = time.perf_counter() if now is None else now

    def increment(self) -> None:
        if self.state is RequestState.FINISHED:
            raise RuntimeError(f"Cannot increment request in state={self.state}")
        self.state = RequestState.INCREMENT

    def finish(self, reason: FinishReason, now: float | None = None) -> None:
        self.state = RequestState.FINISHED
        self.finish_reason = reason
        if self.metrics.finished_at is None:
            self.metrics.finished_at = time.perf_counter() if now is None else now

    def append_token(self, token_id: int) -> None:
        if self.state is RequestState.FINISHED:
            raise RuntimeError("Cannot append token to a finished request")

        self.output_ids.append(int(token_id))

    def record_token(self, token_id: int, now: float | None = None) -> None:
        timestamp = time.perf_counter() if now is None else now
        if self.metrics.first_token_at is None:
            self.metrics.first_token_at = timestamp
        self.append_token(token_id)
        if len(self.output_ids) >= self.sampling.max_new_tokens:
            self.finish(FinishReason.MAX_NEW_TOKENS, now=timestamp)
            return
        if (
            self.sampling.eos_token_id is not None
            and int(token_id) == self.sampling.eos_token_id
        ):
            self.finish(FinishReason.EOS, now=timestamp)


@dataclass(frozen=True, slots=True)
class RequestToken:
    """One streamed token emitted for a request."""

    request: Request
    token_id: int
