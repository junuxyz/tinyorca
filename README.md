# tinyorca

<p align="center">
  <img src="assets/tinyorca-logo.png" alt="tinyorca logo" width="280">
</p>

**tinyorca** is a minimal implementation of an [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf)-style LLM serving engine.

It focuses on iteration-level scheduling and selective batching for mixed prefill and decode workloads.

## Demo: Static Batch vs. Iteration-Level Scheduling

Both demos below use the same setup:

- `max_batch_size=2`
- 5 concurrent requests
- 2 requests(req-0, req-2) intentionally much shorter than the others

### Baseline Engine

<p align="center">
  <img src="assets/baseline_engine_demo.gif" alt="baseline engine demo" width="780">
</p>

In the baseline, the first admitted batch is effectively pinned until its slowest request completes.
Even if one request finishes early, that vacant spot is not turned into useful work right away, so later requests keep waiting.

### tinyorca

<p align="center">
  <img src="assets/tinyorca_demo.gif" alt="tinyorca demo" width="780">
</p>

In tinyorca, scheduling happens at iteration granularity instead of request granularity.

When a short request(e.g. `"Hi"`) finishes, its slot can be reused on the next iteration, so waiting requests can join earlier without waiting for the longest request in the current batch to finish. This helps each step to keep the max batch size, leading to better throughput.

## Deep dive
For a deeper walkthrough of the paper and this implementation, see: **[Understanding Orca through tinyorca](https://github.com/junuxyz/mlsys-notes/blob/main/notes/tinyorca.md)**


## Run

```bash
uv venv
uv sync
uv run python -m tinyorca.example
```

## Example

```python
from tinyorca import OrcaConfig, OrcaServe, SamplingConfig

serve = OrcaServe(
    OrcaConfig(
        model="Qwen/Qwen3-0.6B",
        max_batch_size=2,
        sampling=SamplingConfig(max_new_tokens=32),
    )
)

for event in serve.generate(["Hello", "Hi."]):
    print(event.request.request_id, event.token_id)
```

## Benchmark

```bash
uv run python -m bench
```

By default, the benchmark runs two synthetic workloads:

- `equal_size`: 8 requests of `(128, 128)`
- `short_long_mix`: interleaved short `(32, 32)` and long `(512, 128)` requests

To run just one workload:

```bash
uv run python -m bench --workload short_long_mix
```
