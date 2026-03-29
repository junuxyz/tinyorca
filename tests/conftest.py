from __future__ import annotations

import string

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM


def _build_test_model(vocab_size: int, dtype: torch.dtype | None) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=8,
        )
    ).eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("hf-artifacts") / "tiny-qwen3"

    special_tokens = ["<unk>", "<pad>", "<eos>"]
    vocab = {token: index for index, token in enumerate(special_tokens)}
    for char in string.ascii_letters + string.digits + " <>/?:.-_\n":
        if char not in vocab:
            vocab[char] = len(vocab)

    tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Split(pattern="", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(model_dir)

    model = _build_test_model(len(vocab), torch.float32)
    model.save_pretrained(model_dir)

    return model_dir


@pytest.fixture
def tokenizer(tiny_model_dir):
    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture
def serve_factory(tiny_model_dir):
    from tinyorca.config import OrcaConfig
    from tinyorca.core.serve import OrcaServe

    def factory(
        *,
        estimated_n_slots: int = 32,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **config_overrides,
    ) -> OrcaServe:
        config = OrcaConfig(
            model=str(tiny_model_dir),
            n_slots=estimated_n_slots,
            **config_overrides,
        )
        return OrcaServe(config, device=device, dtype=dtype)

    return factory


@pytest.fixture
def engine_factory(tiny_model_dir):
    from tinyorca.config import OrcaConfig
    from tinyorca.core.engine import OrcaEngine

    def factory(
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **config_overrides,
    ) -> OrcaEngine:
        config = OrcaConfig(model=str(tiny_model_dir), **config_overrides)
        return OrcaEngine(config, device=device, dtype=dtype)

    return factory
