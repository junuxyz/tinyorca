from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import AutoTokenizer

from tinyorca.config import OrcaConfig
from tinyorca.core.engine import OrcaEngine
from tinyorca.core.request import Request, RequestToken, SamplingConfig
from tinyorca.core.scheduler import OrcaScheduler, RequestPool


class Endpoint:
    """Endpoint for input prompt, request construction, and enqueue."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        request_pool: RequestPool,
    ):
        self.tokenizer = tokenizer
        self.request_pool = request_pool
        self._next_request_index = 0

    def submit(
        self,
        prompt_text: str,
        sampling: SamplingConfig,
    ) -> Request:
        request_id = f"req-{self._next_request_index}"
        self._next_request_index += 1

        prompt_ids = tuple(
            self.tokenizer.encode(
                prompt_text,
                add_special_tokens=False,
                verbose=False,
            )
        )
        request = Request(request_id, prompt_ids, sampling)
        request.mark_submitted()
        self.request_pool.push(request)
        return request


class OrcaServe:
    """Minimal ORCA serving stack with a streaming generate API."""

    def __init__(
        self,
        config: OrcaConfig,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if not config.model:
            raise ValueError("config.model must be set")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = eos_token_id
        self.request_pool = RequestPool()
        self.sampling = SamplingConfig(
            max_new_tokens=config.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        self.endpoint = Endpoint(self.tokenizer, self.request_pool)
        self.engine = OrcaEngine(config=config, device=device, dtype=dtype)
        self.scheduler = OrcaScheduler(
            self.engine, self.request_pool, config.max_batch_size
        )

    def generate(
        self,
        prompt_texts: list[str],
        sampling: SamplingConfig | list[SamplingConfig] | None = None,
    ) -> Iterator[RequestToken]:
        """Submit a prompt batch and stream token events until the scheduler drains."""
        if sampling is None:
            samplings = [None] * len(prompt_texts)
        elif isinstance(sampling, SamplingConfig):
            samplings = [sampling] * len(prompt_texts)
        else:
            samplings = sampling
        if len(samplings) != len(prompt_texts):
            raise ValueError("sampling must match the number of prompts")
        for prompt_text, sampling in zip(prompt_texts, samplings, strict=True):
            if sampling is None:
                sampling = self.sampling
            elif sampling.eos_token_id is None:
                sampling = SamplingConfig(
                    max_new_tokens=sampling.max_new_tokens,
                    eos_token_id=self.sampling.eos_token_id,
                )
            self.endpoint.submit(prompt_text=prompt_text, sampling=sampling)
        yield from self.scheduler.schedule()
