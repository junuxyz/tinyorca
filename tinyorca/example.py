from __future__ import annotations

import sys
import time

from rich.live import Live

from tinyorca import OrcaConfig, OrcaServe, SamplingConfig

THINKING_FRAMES = ("Thinking", "Thinking .", "Thinking ..", "Thinking ...")
THINKING_STEP_SECONDS = 0.3


def render_stream(serve: OrcaServe, prompts: list[str]) -> None:
    tokenizer = serve.tokenizer
    base_index = serve.endpoint._next_request_index
    short_request_ids = {
        f"req-{base_index + i}" for i, prompt in enumerate(prompts) if len(prompt.split()) <= 2
    }
    if tokenizer.chat_template is None:
        formatted_prompts = prompts
    else:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    stream = serve.generate(formatted_prompts)
    requests_by_id = {}
    started_at = time.monotonic()

    if sys.stdout.isatty():
        with Live("", auto_refresh=False) as live:
            for event in stream:
                requests_by_id[event.request.request_id] = event.request
                thinking = THINKING_FRAMES[
                    int((time.monotonic() - started_at) / THINKING_STEP_SECONDS)
                    % len(THINKING_FRAMES)
                ]
                lines = []
                for request in requests_by_id.values():
                    text = tokenizer.decode(request.output_ids, skip_special_tokens=True)
                    if "</think>" in text:
                        text = text.partition("</think>")[2].lstrip()
                    elif "<think>" in text or not text:
                        text = thinking
                    short = " (short request)" if request.request_id in short_request_ids else ""
                    lines.append(f"{request.request_id}{short}: {text}")
                live.update(
                    "\n".join(lines),
                    refresh=True,
                )
        return

    previous_text_by_id: dict[str, str] = {}
    for event in stream:
        request = event.request
        request_id = request.request_id
        current_text = tokenizer.decode(request.output_ids, skip_special_tokens=True)
        if "</think>" in current_text:
            current_text = current_text.partition("</think>")[2].lstrip()
        elif "<think>" in current_text or not current_text:
            current_text = THINKING_FRAMES[
                int((time.monotonic() - started_at) / THINKING_STEP_SECONDS)
                % len(THINKING_FRAMES)
            ]
        previous_text = previous_text_by_id.get(request_id, "")
        if current_text.startswith(previous_text):
            delta = current_text[len(previous_text) :]
        else:
            delta = current_text
        if delta:
            short = " (short request)" if request_id in short_request_ids else ""
            print(f"{request_id}{short}: {delta}")
        previous_text_by_id[request_id] = current_text


def main() -> None:
    serve = OrcaServe(
        OrcaConfig(
            model="Qwen/Qwen3-0.6B",
            max_batch_size=2,
            sampling=SamplingConfig(max_new_tokens=512),
        )
    )

    render_stream(
        serve,
        [
            "Hi",  # (short request)
            "Explain what Orca is.",
            "Hi",  # (short request)
            "Give a simple pasta idea.",
            "Explain me about quantum physics.",
        ],
    )


if __name__ == "__main__":
    main()
