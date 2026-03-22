import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import torch

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from src.eval.generation import (
        build_freeform_prompt,
        infer_model_mode,
        list_checkpoint_options,
        load_generation_resources,
        prepare_prompt_text,
    )
    from src.store import RunStore
    from src.utils.device import get_device

    run_options = [run.run_id for run in RunStore().list_runs()] or ["none"]

    return (
        build_freeform_prompt,
        get_device,
        infer_model_mode,
        list_checkpoint_options,
        load_generation_resources,
        mo,
        prepare_prompt_text,
        run_options,
        torch,
    )


@app.cell
def _(mo, run_options):
    run_id = mo.ui.dropdown(options=run_options, value=run_options[0], label="Run")
    chat_mode = mo.ui.dropdown(
        options=["story", "chat"],
        value="story",
        label="Mode",
    )
    temperature = mo.ui.slider(
        start=0.0,
        stop=1.5,
        step=0.1,
        value=0.8,
        label="Temperature",
    )
    num_tokens = mo.ui.number(value=96, start=8, stop=256, label="Max new tokens")
    max_turns = mo.ui.number(value=6, start=1, stop=24, label="Turns of history")
    sampler_name = mo.ui.dropdown(
        options=["confidence_iterative", "full_refresh"],
        value="confidence_iterative",
        label="Sampler",
    )
    debug_trace = mo.ui.checkbox(value=False, label="Show debug trace")
    return (
        chat_mode,
        debug_trace,
        max_turns,
        num_tokens,
        run_id,
        sampler_name,
        temperature,
    )


@app.cell
def _(list_checkpoint_options, mo, run_id):
    checkpoint_options = (
        list_checkpoint_options(run_id.value) if run_id.value != "none" else []
    )
    checkpoint = mo.ui.dropdown(
        options=checkpoint_options or ["none"],
        value=checkpoint_options[0] if checkpoint_options else "none",
        label="Checkpoint",
    )
    checkpoint  # noqa: B018
    return checkpoint, checkpoint_options


@app.cell
def _(
    chat_mode,
    checkpoint,
    get_device,
    infer_model_mode,
    load_generation_resources,
    mo,
    run_id,
):
    system_prompt = mo.ui.text_area(
        value="Continue the exchange in a clear, useful tone.",
        label="System prompt",
        rows=3,
    )
    if run_id.value == "none" or checkpoint.value == "none":
        model_details = mo.callout("No run is available yet.", kind="warn")
        model_kind = "story_base"
    else:
        resources = load_generation_resources(
            run_id.value,
            checkpoint.value,
            get_device(),
        )
        model_kind = infer_model_mode(resources.config)
        warning = None
        if chat_mode.value == "chat" and model_kind == "story_base":
            warning = (
                "This checkpoint is TinyStories-style only. Chat mode is "
                "out-of-distribution and may degrade badly."
            )
        rows = [
            f"- Model type: `{model_kind}`",
            f"- Dataset format: `{resources.config.dataset.format_name}`",
            f"- Sampler: `{resources.config.diffusion.sampler_name}`",
            f"- Context window: `{resources.config.model.max_seq_len}` tokens",
        ]
        model_details = mo.vstack(
            [
                mo.md("\n".join(rows)),
                (
                    mo.callout(warning, kind="warn")
                    if warning
                    else mo.md("Sampler trace can be expanded below.")
                ),
            ],
            gap=1,
        )
    mo.vstack([system_prompt, model_details], gap=1)
    return model_kind, system_prompt


@app.cell
def _(
    chat_mode,
    debug_trace,
    get_device,
    load_generation_resources,
    max_turns,
    mo,
    num_tokens,
    prepare_prompt_text,
    sampler_name,
    run_id,
    system_prompt,
    temperature,
    checkpoint,
    torch,
):
    def local_diffusion_chat(messages, config):
        del config

        if run_id.value == "none":
            yield "No run is available yet. Train a model first."
            return

        try:
            resources = load_generation_resources(
                run_id.value,
                checkpoint.value if checkpoint.value != "none" else None,
                get_device(),
            )
        except Exception as exc:
            yield f"Unable to load the selected run: {exc}"
            return

        prompt_text = prepare_prompt_text(
            chat_mode.value,
            messages=messages,
            system_prompt=system_prompt.value,
            max_turns=int(max_turns.value),
        )
        available_context = resources.config.model.max_seq_len - int(num_tokens.value)
        if available_context <= 0:
            yield (
                "Reduce `Max new tokens`; it must leave room in the model "
                f"context window of {resources.config.model.max_seq_len} tokens."
            )
            return

        prompt_text = prompt_text[-available_context:]
        prompt_ids = resources.tokenizer.encode(prompt_text)
        prompt_ids = prompt_ids[-available_context:]
        prompt_tensor = (
            None if not prompt_ids else torch.tensor([prompt_ids], dtype=torch.long)
        )
        prompt_len = len(prompt_ids)

        final_tokens = None
        trace_lines: list[str] = []
        for step in resources.sampler.sample(
            prompt_tokens=prompt_tensor,
            num_tokens=int(num_tokens.value),
            temperature=float(temperature.value),
            sampler_name=sampler_name.value,
        ):
            final_tokens = step.tokens
            if debug_trace.value:
                trace_lines.append(
                    "step="
                    f"{step.step_index} t={step.timestep} revealed={step.num_revealed} "
                    f"remaining={step.num_masked_remaining}"
                )

        if final_tokens is None:
            yield "Generation failed before producing any tokens."
            return

        response_ids = final_tokens[0, prompt_len:].tolist()
        response_text = resources.tokenizer.decode(response_ids).strip()
        if not response_text:
            response_text = "(empty response)"
        if debug_trace.value and trace_lines:
            response_text += "\n\n[trace]\n" + "\n".join(trace_lines[:12])
        yield response_text

    chat = mo.ui.chat(
        local_diffusion_chat,
        prompts=[
            "Write a short bedtime story about a paper boat.",
            "Continue: The robot opened the attic door and found...",
            "Help me brainstorm a cozy campfire scene.",
        ],
    )
    chat  # noqa: B018
    return (chat,)


@app.cell
def _(chat, mo):
    messages = [
        {
            "role": str(getattr(message, "role", "")),
            "content": str(getattr(message, "content", "")),
        }
        for message in chat.value
    ]
    history = mo.ui.table(messages) if messages else mo.md("No conversation yet.")
    mo.vstack([mo.md("## Conversation log"), history], gap=1)
    return


if __name__ == "__main__":
    app.run()
