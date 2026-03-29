# tinyORCA notes

## Architecture Overview

<p align="center">
  <img
    src="assets/tinyorca-architecture.png"
    alt="tinyORCA architecture overview"
    width="960"
  >
</p>

`generate()` submits prompts through Endpoint. Then OrcaScheduler repeatedly selects an arrival-ordered batch from RequestPool. OrcaEngine runs one iteration of mixed prefill/decode. Each iteration yields streamed token events for that batch.

## Runtime Structure

`OrcaServe` has four submodules:

- `Endpoint`: tokenizes prompt text, builds `Request`, and enqueues it
- `RequestPool`: stores all non-finished requests
- `OrcaScheduler`: owns selection and admission control (control plane)
- `OrcaEngine`: owns model execution, KV cache state, and token generation (execution plane)

Request state follows one small lifecycle:

```text
INITIATION (prefill) -> INCREMENT (decode) -> FINISHED
```

`RequestPool.arrival_ordered_requests()` always returns requests in submit order.

That preserves the paper's iteration-level FCFS rule:

- earlier-arrived requests never fall behind later-arrived requests in completed iterations
- `INITIATION` requests still pay the one-time KV reservation check before first admission

## Iterative Scheduling

The scheduler is based on Algorithm 1 presented from the paper:

<p align="center">
  <img
    src="assets/orca-scheduling-algorithm-paper.png"
    alt="ORCA scheduling algorithm from the paper"
    width="400"
  >
</p>


Scheduler has three important limits:

- `max_batch_size`: the primary cap on how many requests run in one iteration
- `n_slots`: the hard ceiling on total reserved KV slots
- `n_rsrv`: slots already reserved by admitted requests

`n_slots` is estimated once from CUDA memory when the scheduler is created, then treated as a hard ceiling.

For a request in `INITIATION`, the scheduler reserves:

```text
len(prompt_ids) + request.sampling.max_new_tokens
```

`select()` does three things:

1. takes requests in arrival order
2. stops at `max_batch_size`
3. admits new `INITIATION` requests only if their reservation fits inside `n_slots`


**Upper bounds**
- if one request is larger than total `n_slots`, selection raises an explicit error
- if `n_rsrv + request_slots > n_slots`, the scheduler stops admitting more new requests for that iteration

`schedule()` is the outer loop:

1. call `select()`
2. run one engine iteration
3. remove finished requests and release their reservation
4. otherwise move them back to `INCREMENT`
5. stream emitted `RequestToken` events

## Selective Batching

<p align="center">
  <img
    src="assets/orca-selective-batching-paper.png"
    alt="ORCA selective batching from the paper"
    width="750"
  >
</p>

`OrcaEngine.run_iter(requests)` runs exactly one model step for the selected batch and returns at most one token per request.

One iteration can mix prefill and decode. 

`run_iter` works as follows:

1. If `request.output_ids` is empty, this step is prefill and it feeds the full prompt. Otherwise, this step is decode and it feeds only the last generated token.

2. `build_flat_batch()` flattens all step tokens into one tensor and keeps per-request metadata:
   - `RequestSpan` for where each request lives in the flat tensor
   - `position_ids`
   - `cache_position`

3. The selective Qwen3 wrapper uses that metadata to run attention request **by request** while keeping the rest of the layer flow flat.

4. After the forward pass, the engine takes the last hidden state for each request span, projects with `lm_head`, picks `argmax`, records the token, and drops that request's cache if it finished. The scheduler then removes finished requests and releases their reservations.


## Limitation

Selective attention is currently implemented in Python on top of Hugging Face Qwen3 internals. That keeps the code readable, but it is much slower than a production ORCA-style system with specialized kernels or prior FasterTransformer-like engine (check benchmark below).

## Benchmark

`bench.py` only measures one fixed synthetic workload (128, 128) due to my tiny VRAM(4GB) size :(

```bash
python -m bench
```

Benchmark reports elapsed time, token throughput, TTFT, TPOT, and E2E latency for that fixed case.


### tinyorca result
```bash
❯ python -m bench
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████| 311/311 [00:00<00:00, 1693.61it/s]
Estimated n_slots from GPU memory: n_slots=8865
kv_slot_bytes=114688
activation_peak_bytes=16261120
Qwen/Qwen3-0.6B
cuda / bfloat16
workload: 128x128 (128, 128)

field             value
----------------  ------
requests          8
warmup            2
batch             4
elapsed_s         50.341
requests_per_s    0.159
input_tok_per_s   20.341
output_tok_per_s  20.341
total_tok_per_s   40.683
input_tokens      1024
output_tokens     1024
total_tokens      2048

latency_ms  mean      p50       p95       p99
----------  --------  --------  --------  --------
ttft        12661.71  12661.86  25143.52  25143.70
tpot        196.74    196.74    198.38    198.38
e2e         37648.26  37648.41  50337.16  50337.34
```

### baseline engine result
```bash
❯ python -m labs.bench.bench
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 311/311 [00:00<00:00, 1353.95it/s]
Qwen/Qwen3-0.6B
cuda / bfloat16
workload: 128x128 (128, 128)

field             value
----------------  -------
requests          8
warmup            2
batch             4
elapsed_s         15.962
requests_per_s    0.501
input_tok_per_s   64.152
output_tok_per_s  64.152
total_tok_per_s   128.304
input_tokens      1024
output_tokens     1024
total_tokens      2048

latency_ms  mean      p50       p95       p99
----------  --------  --------  --------  --------
ttft        4156.84   4156.86   8196.55   8196.69
tpot        61.95     61.95     62.74     62.74
e2e         12023.99  12024.01  15962.69  15962.84
```

## Related Reading

- ORCA paper
