# tinyORCA

<p align="center">
  <img src="assets/tinyorca-logo.png" alt="tinyORCA logo" width="320">
</p>

`tinyORCA` is a small, readable reimplementation of the main ORCA scheduling idea.

It is built for understanding, not production.

- iteration-level scheduling
- iteration-level FCFS
- selective batching
- hard `n_slots` admission control

<p align="center">
  <img src="assets/tinyorca_demo.gif" alt="tinyORCA demo" width="780">
</p>

As you can see in the demo, it continuously batches. This is available because of selective batching. Selective batching enables to schedule mixture of prefill and decode requests.

## Install

```bash
uv venv
uv sync
```

## Minimal Example

```python
from tinyorca import OrcaConfig, OrcaServe

serve = OrcaServe(
    OrcaConfig(
        model="Qwen/Qwen3-0.6B",
        max_batch_size=2,
        max_new_tokens=32,
    )
)

for event in serve.generate(["Hello", "Hi."]):
    print(event.request.request_id, event.token_id)
```

See [example.py](/tinyorca/example.py) for a slightly fuller runnable example.

## Scope

`tinyORCA` currently stays narrow on purpose:

- single-process
- Qwen3-based selective attention path
- educational implementation over optimized kernels

## More Detail

Architecture notes, scheduling explanation, and benchmark notes live in [note.md](/note.md).
