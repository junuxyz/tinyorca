from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OrcaConfig:
    """Shared runtime configuration for tinyORCA submodules."""

    model: str | None = None
    max_batch_size: int = 4
    max_new_tokens: int = 64
    n_slots: int | None = None
    gpu_utilization: float = 0.8

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.n_slots is not None and self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if not 0.0 < self.gpu_utilization <= 1.0:
            raise ValueError("gpu_utilization must be in (0, 1]")
