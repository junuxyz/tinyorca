from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from tinyorca.config import OrcaConfig
from tinyorca.core.request import Request, RequestState, RequestToken, SamplingConfig
from tinyorca.models.qwen3 import Qwen3SelectiveModel, RequestSpan


@dataclass(slots=True)
class FlatBatch:
    """
    Selective batching view of a mixed prefill/decode batch.
    """

    hidden_states: torch.Tensor  # [sum_S, Hidden]
    spans: list[RequestSpan]
    position_ids: list[torch.Tensor] = field(default_factory=list)
    cache_position: list[torch.Tensor] = field(default_factory=list)


class OrcaEngine:
    """
    Execution engine for one batched model iteration.
    Supports selective batching.
    """

    def __init__(
        self,
        config: OrcaConfig,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if not config.model:
            raise ValueError("config.model must be set")

        self.config = config
        model_name = config.model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
        self.device = resolved_device
        hf_config = AutoConfig.from_pretrained(model_name)
        hf_config.tie_word_embeddings = False
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=hf_config,
            dtype=dtype,
        ).to(self.device)
        self.hf_model.eval()
        self.model = Qwen3SelectiveModel(self.hf_model)
        self.request_caches: dict[str, DynamicCache] = {}

    @property
    def kv_slot_bytes(self) -> int:
        param_dtype = next(self.hf_model.parameters()).dtype
        dtype_size = torch.tensor([], dtype=param_dtype).element_size()
        num_hidden_layers = int(self.hf_model.config.num_hidden_layers)
        num_kv_heads = int(self.hf_model.config.num_key_value_heads)
        head_dim = int(
            getattr(
                self.hf_model.config,
                "head_dim",
                self.hf_model.config.hidden_size
                // self.hf_model.config.num_attention_heads,
            )
        )
        return 2 * num_hidden_layers * num_kv_heads * head_dim * dtype_size

    def reset_cache_state(self) -> None:
        self.request_caches.clear()

    def estimate_activation_peak_bytes(
        self,
        max_batch_size: int,
        max_new_tokens: int,
    ) -> int:
        """Warm up one batch shape and measure the temporary activation peak."""
        warmup_requests = [
            Request(
                request_id=f"warmup-{index}",
                prompt_ids=tuple([0] * max_new_tokens),
                sampling=SamplingConfig(
                    max_new_tokens=max_new_tokens,
                    eos_token_id=None,
                ),
            )
            for index in range(max_batch_size)
        ]

        self.reset_cache_state()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        self.run_iter(warmup_requests)
        memory_stats = torch.cuda.memory_stats(self.device)
        activation_peak_bytes = (
            memory_stats["allocated_bytes.all.peak"]
            - memory_stats["allocated_bytes.all.current"]
        )

        self.reset_cache_state()
        torch.cuda.empty_cache()

        return max(0, int(activation_peak_bytes))

    def estimate_n_slots(
        self,
        max_batch_size: int,
        max_new_tokens: int,
    ) -> int:
        """Estimate a one-time hard slot ceiling from current CUDA memory."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("estimate_n_slots requires CUDA")

        activation_peak_bytes = self.estimate_activation_peak_bytes(
            max_batch_size=max_batch_size,
            max_new_tokens=max_new_tokens,
        )
        free, total = torch.cuda.mem_get_info()
        used = total - free
        usable_bytes = int(total * self.config.gpu_utilization)
        available_slot_bytes = usable_bytes - used - activation_peak_bytes
        n_slots = available_slot_bytes // self.kv_slot_bytes
        print(
            "Estimated n_slots from GPU memory: "
            f"n_slots={n_slots}\n"
            f"kv_slot_bytes={self.kv_slot_bytes}\n"
            f"activation_peak_bytes={activation_peak_bytes}",
            flush=True,
        )
        if n_slots < 1:
            raise ValueError(
                "Estimated n_slots must be >= 1 "
                f"(activation_peak_bytes={activation_peak_bytes}, "
                f"usable_bytes={usable_bytes}, used={used})"
            )
        return int(n_slots)

    def run_iter(
        self,
        requests: list[Request],
    ) -> list[RequestToken]:
        """
        One selective-batching iteration.
        """
        if not requests:
            raise ValueError("run_iter requires at least one request")

        flat_batch = self.build_flat_batch(requests)

        with torch.inference_mode():
            output_hidden_states = self.model(
                hidden_states=flat_batch.hidden_states,
                spans=flat_batch.spans,
                position_ids=flat_batch.position_ids,
                cache_position=flat_batch.cache_position,
                request_caches=self.request_caches,
            )

            # one iteration emits at most one token event per selected request.
            step_time = time.perf_counter()
            token_events: list[RequestToken] = []
            last_hidden_states = torch.stack(
                [output_hidden_states[span.end - 1] for span in flat_batch.spans]
            )
            next_token_ids = torch.argmax(
                self.hf_model.lm_head(last_hidden_states),
                dim=-1,
            ).tolist()

            for request, next_token_id in zip(requests, next_token_ids, strict=True):
                request.record_token(next_token_id, now=step_time)
                token_events.append(RequestToken(request, next_token_id))

                if request.state is RequestState.FINISHED:
                    self.request_caches.pop(request.request_id, None)

        return token_events

    def build_flat_batch(self, requests: list[Request]) -> FlatBatch:
        input_token_ids: list[int] = []
        spans: list[RequestSpan] = []
        position_ids: list[torch.Tensor] = []
        cache_position: list[torch.Tensor] = []

        flat_start = 0
        for request in requests:
            cache = self.request_caches.get(request.request_id)
            if cache is None:
                # one HF DynamicCache holds the full per-request KV state.
                #
                # for Qwen3-0.6B, structure of cache is:
                #
                #   DynamicCache(
                #     layers=[
                #       layer0: {keys: [1, 8, seq_len, 128], values: [1, 8, seq_len, 128]},
                #       layer1: {keys: [1, 8, seq_len, 128], values: [1, 8, seq_len, 128]},
                #       ...
                #       layer27: {keys: [1, 8, seq_len, 128], values: [1, 8, seq_len, 128]},
                #     ]
                #   )
                #
                # where `seq_len` grows from prompt_len to prompt_len + generated_tokens.
                # creating this object only prepares the empty container.
                # The actual KV tensors are filled later during the Qwen3
                # attention forward pass in models/qwen3.py.
                cache = DynamicCache(config=self.hf_model.config)
                self.request_caches[request.request_id] = cache

            # prefill
            if not request.output_ids:
                step_token_ids = request.prompt_ids
                processed_tokens = 0
            # decode
            else:
                step_token_ids = (request.output_ids[-1],)
                processed_tokens = len(request.prompt_ids) + len(request.output_ids) - 1

            step_len = len(step_token_ids)  # prefill: len(prompt_ids); decode: 1
            total_visible_tokens = processed_tokens + step_len

            # span info is later used to split per request
            spans.append(
                RequestSpan(
                    request_id=request.request_id,
                    start=flat_start,
                    end=flat_start + step_len,
                )
            )
            # absolute token positions for this step, used for RoPE.
            request_position_ids = torch.arange(
                processed_tokens,
                total_visible_tokens,
                device=self.device,
                dtype=torch.long,
            )
            position_ids.append(request_position_ids)
            cache_position.append(request_position_ids)

            input_token_ids.extend(step_token_ids)
            flat_start += step_len

        input_ids = torch.tensor(input_token_ids, device=self.device, dtype=torch.long)
        hidden_states = self.hf_model.model.embed_tokens(input_ids)

        return FlatBatch(
            hidden_states=hidden_states,
            spans=spans,
            position_ids=position_ids,
            cache_position=cache_position,
        )
