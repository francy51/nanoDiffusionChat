import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    for _candidate in (Path.cwd(), *Path.cwd().parents):
        if (_candidate / "pyproject.toml").exists() and (_candidate / "src").exists():
            candidate_str = str(_candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from src.eval.generation import (
        generate_candidates,
        infer_model_mode,
        list_checkpoint_options,
        load_generation_resources,
    )
    from src.eval.manual import (
        append_manual_eval_record,
        build_manual_eval_record,
        summarize_manual_eval_records,
    )
    from src.store import RunStore
    from src.utils.device import get_device

    run_options = [run.run_id for run in RunStore().list_runs()] or ["none"]
    return (
        append_manual_eval_record,
        build_manual_eval_record,
        generate_candidates,
        get_device,
        infer_model_mode,
        list_checkpoint_options,
        load_generation_resources,
        mo,
        run_options,
        summarize_manual_eval_records,
    )


@app.cell
def _(mo, run_options):
    run_id = mo.ui.dropdown(options=run_options, value=run_options[0], label="Run")
    compare_run_id = mo.ui.dropdown(
        options=["none", *run_options],
        value="none",
        label="Compare run",
    )
    mode = mo.ui.dropdown(
        options=["story", "chat", "freeform"],
        value="story",
        label="Mode",
    )
    sampler_name = mo.ui.dropdown(
        options=["confidence_iterative", "full_refresh"],
        value="confidence_iterative",
        label="Sampler",
    )
    num_tokens = mo.ui.number(value=96, start=8, stop=256, label="Max new tokens")
    temperature = mo.ui.slider(
        start=0.0,
        stop=1.5,
        step=0.1,
        value=0.8,
        label="Temperature",
    )
    seed = mo.ui.number(value=7, start=0, stop=100_000, label="Seed")
    candidate_count = mo.ui.number(value=2, start=1, stop=4, label="Candidates")
    generate_action = mo.ui.run_button(label="Generate")
    save_action = mo.ui.run_button(label="Save eval")
    controls = mo.vstack(
        [
            mo.md("# Manual Eval Lab"),
            run_id,
            compare_run_id,
            mode,
            sampler_name,
            num_tokens,
            temperature,
            seed,
            candidate_count,
            mo.hstack([generate_action, save_action]),
        ],
        gap=1,
    )
    controls  # noqa: B018
    return (
        candidate_count,
        compare_run_id,
        generate_action,
        mode,
        num_tokens,
        run_id,
        sampler_name,
        save_action,
        seed,
        temperature,
    )


@app.cell
def _(compare_run_id, list_checkpoint_options, mo, run_id):
    checkpoint_options = (
        list_checkpoint_options(run_id.value) if run_id.value != "none" else []
    )
    checkpoint = mo.ui.dropdown(
        options=checkpoint_options or ["none"],
        value=checkpoint_options[0] if checkpoint_options else "none",
        label="Checkpoint",
    )

    compare_checkpoint_options = (
        list_checkpoint_options(compare_run_id.value)
        if compare_run_id.value not in {"none", run_id.value}
        else []
    )
    compare_checkpoint = mo.ui.dropdown(
        options=["none", *compare_checkpoint_options],
        value="none",
        label="Compare checkpoint",
    )
    mo.hstack([checkpoint, compare_checkpoint], widths=[1, 1])
    return checkpoint, compare_checkpoint


@app.cell
def _(mo):
    prompt_text = mo.ui.text_area(
        value="Write a short bedtime story about a paper boat.",
        label="Prompt",
        rows=6,
    )
    system_prompt = mo.ui.text_area(
        value="Respond clearly and stay grounded in the prompt.",
        label="System prompt",
        rows=4,
    )
    prompt_presets = mo.ui.dropdown(
        options=[
            "Write a short bedtime story about a paper boat.",
            "Explain the moon to a five-year-old.",
            "Continue: The attic door opened and the room smelled like rain.",
        ],
        value="Write a short bedtime story about a paper boat.",
        label="Preset",
    )
    notes = mo.ui.text_area(label="Evaluator notes", rows=6)
    mo.vstack([prompt_presets, prompt_text, system_prompt, notes], gap=1)
    return notes, prompt_presets, prompt_text, system_prompt


@app.cell
def _(prompt_presets, prompt_text):
    if prompt_text.value.strip() == "":
        prompt_value = prompt_presets.value
    else:
        prompt_value = prompt_text.value
    return (prompt_value,)


@app.cell
def _(
    candidate_count,
    checkpoint,
    compare_checkpoint,
    compare_run_id,
    generate_action,
    generate_candidates,
    get_device,
    load_generation_resources,
    mode,
    num_tokens,
    prompt_value,
    run_id,
    sampler_name,
    seed,
    system_prompt,
    temperature,
):
    primary_candidates = []
    comparison_candidates = []
    primary_resources = None
    comparison_resources = None

    if (
        generate_action.value
        and run_id.value != "none"
        and checkpoint.value != "none"
        and prompt_value.strip()
    ):
        primary_resources = load_generation_resources(
            run_id.value,
            checkpoint.value,
            get_device(),
        )
        primary_candidates = generate_candidates(
            primary_resources,
            prompt_text=prompt_value,
            num_new_tokens=int(num_tokens.value),
            temperature=float(temperature.value),
            candidate_count=int(candidate_count.value),
            seed=int(seed.value),
            sampler_name=sampler_name.value,
        )

        if compare_run_id.value != "none" and compare_checkpoint.value != "none":
            comparison_resources = load_generation_resources(
                compare_run_id.value,
                compare_checkpoint.value,
                get_device(),
            )
            comparison_candidates = generate_candidates(
                comparison_resources,
                prompt_text=prompt_value,
                num_new_tokens=int(num_tokens.value),
                temperature=float(temperature.value),
                candidate_count=int(candidate_count.value),
                seed=int(seed.value),
                sampler_name=sampler_name.value,
            )

    return (
        comparison_candidates,
        comparison_resources,
        primary_candidates,
        primary_resources,
    )


@app.cell
def _(candidate_count, mode, mo):
    score_widgets = []
    flag_widgets = []
    for candidate_index in range(int(candidate_count.value)):
        coherence = mo.ui.number(
            value=3, start=1, stop=5, label=f"C{candidate_index + 1} coherence"
        )
        fluency = mo.ui.number(
            value=3, start=1, stop=5, label=f"C{candidate_index + 1} fluency"
        )
        relevance = mo.ui.number(
            value=3, start=1, stop=5, label=f"C{candidate_index + 1} relevance"
        )
        following = mo.ui.number(
            value=3,
            start=1,
            stop=5,
            label=f"C{candidate_index + 1} instruction",
        )
        special = mo.ui.number(
            value=3,
            start=1,
            stop=5,
            label=(
                f"C{candidate_index + 1} story"
                if mode.value == "story"
                else f"C{candidate_index + 1} chat"
            ),
        )
        score_widgets.append(
            {
                "coherence": coherence,
                "fluency": fluency,
                "relevance": relevance,
                "instruction_following": following,
                (
                    "story_quality" if mode.value == "story" else "chat_naturalness"
                ): special,
            }
        )
        flag_widgets.append(
            {
                "garbage_text": mo.ui.checkbox(
                    value=False, label=f"C{candidate_index + 1} garbage"
                ),
                "repetition_loop": mo.ui.checkbox(
                    value=False, label=f"C{candidate_index + 1} repetition"
                ),
                "off_prompt": mo.ui.checkbox(
                    value=False, label=f"C{candidate_index + 1} off prompt"
                ),
                "cut_off": mo.ui.checkbox(
                    value=False, label=f"C{candidate_index + 1} cut off"
                ),
                "undesirable_content": mo.ui.checkbox(
                    value=False, label=f"C{candidate_index + 1} undesirable"
                ),
            }
        )
    return flag_widgets, score_widgets


@app.cell
def _(
    comparison_candidates,
    infer_model_mode,
    mo,
    primary_candidates,
    primary_resources,
    score_widgets,
):
    if not primary_candidates or primary_resources is None:
        outputs = mo.callout(
            "Press `Generate` to create candidates for scoring.",
            kind="warn",
        )
    else:
        sections = []
        model_type = infer_model_mode(primary_resources.config)
        sections.append(mo.md(f"## Primary checkpoint\nModel type: `{model_type}`"))
        for _candidate, widgets in zip(primary_candidates, score_widgets, strict=False):
            sections.append(
                mo.vstack(
                    [
                        mo.md(
                            f"### Candidate {_candidate.candidate_index + 1}\n\n"
                            f"```\n{_candidate.text or '(empty response)'}\n```"
                        ),
                        mo.hstack(list(widgets.values()), widths="equal"),
                        (
                            mo.ui.table(_candidate.trace_rows)
                            if _candidate.trace_rows
                            else mo.md("No sampler trace.")
                        ),
                    ],
                    gap=1,
                )
            )
        if comparison_candidates:
            sections.append(mo.md("## Comparison checkpoint"))
            for _candidate in comparison_candidates:
                sections.append(
                    mo.md(
                        f"### Comparison {_candidate.candidate_index + 1}\n\n"
                        f"```\n{_candidate.text or '(empty response)'}\n```"
                    )
                )
        outputs = mo.vstack(sections, gap=1)
    outputs  # noqa: B018
    return


@app.cell
def _(flag_widgets, mo, score_widgets):
    scoring_panels = []
    for index, (_scores, _flags) in enumerate(
        zip(score_widgets, flag_widgets, strict=False)
    ):
        scoring_panels.append(
            mo.vstack(
                [
                    mo.md(f"### Candidate {index + 1} rubric"),
                    mo.hstack(list(_scores.values()), widths="equal"),
                    mo.hstack(list(_flags.values()), widths="equal"),
                ],
                gap=1,
            )
        )
    sidebar = (
        mo.vstack([mo.md("## Rubric"), *scoring_panels], gap=1)
        if scoring_panels
        else mo.md("No scoring controls yet.")
    )
    sidebar  # noqa: B018
    return


@app.cell
def _(
    append_manual_eval_record,
    build_manual_eval_record,
    checkpoint,
    compare_checkpoint,
    comparison_resources,
    compare_run_id,
    flag_widgets,
    mode,
    notes,
    num_tokens,
    primary_candidates,
    primary_resources,
    prompt_value,
    run_id,
    sampler_name,
    save_action,
    score_widgets,
    system_prompt,
    temperature,
):
    save_status = "Press `Save eval` after generating candidates."
    if save_action.value and primary_candidates and primary_resources is not None:
        generation_params = {
            "sampler_name": sampler_name.value,
            "num_new_tokens": int(num_tokens.value),
            "temperature": float(temperature.value),
        }
        for _candidate, _scores, _flags in zip(
            primary_candidates,
            score_widgets,
            flag_widgets,
            strict=False,
        ):
            rubric_scores = {key: int(widget.value) for key, widget in _scores.items()}
            failure_flags = {key: bool(widget.value) for key, widget in _flags.items()}
            record = build_manual_eval_record(
                run_id=run_id.value,
                checkpoint_path=str(primary_resources.checkpoint_path),
                comparison_checkpoint_path=(
                    str(comparison_resources.checkpoint_path)
                    if comparison_resources is not None
                    else None
                ),
                mode=mode.value,
                prompt_text=prompt_value,
                system_prompt=system_prompt.value,
                generation_params=generation_params,
                candidate_index=_candidate.candidate_index,
                generated_text=_candidate.text,
                rubric_scores=rubric_scores,
                failure_flags=failure_flags,
                evaluator_notes=notes.value,
            )
            append_manual_eval_record(
                primary_resources.checkpoint_path.parent.parent, record
            )
        save_status = "Saved manual evaluation records."
    return (save_status,)


@app.cell
def _(
    checkpoint,
    compare_checkpoint,
    mo,
    primary_resources,
    save_status,
    summarize_manual_eval_records,
):
    if primary_resources is None:
        summary = mo.md("No manual evaluation summary yet.")
    else:
        summary_rows = summarize_manual_eval_records(
            primary_resources.checkpoint_path.parent.parent
        )
        rows = [{"metric": key, "value": value} for key, value in summary_rows.items()]
        rows.extend(
            [
                {"metric": "Selected checkpoint", "value": checkpoint.value},
                {"metric": "Comparison checkpoint", "value": compare_checkpoint.value},
            ]
        )
        summary = mo.ui.table(rows)
    mo.vstack([mo.md("## Eval summary"), mo.md(save_status), summary], gap=1)
    return


if __name__ == "__main__":
    app.run()
