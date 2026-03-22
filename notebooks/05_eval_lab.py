import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from src.store import RunStore as _RunStore

    run_options = [run.run_id for run in _RunStore().list_runs()] or ["none"]
    run_id = mo.ui.dropdown(options=run_options, value=run_options[0], label="Run")
    num_batches = mo.ui.number(value=4, start=1, stop=128, label="Eval batches")
    max_examples = mo.ui.number(
        value=12,
        start=1,
        stop=64,
        label="Decoded examples",
    )
    action = mo.ui.run_button(label="Evaluate")
    controls = mo.vstack(
        [
            mo.md("## Evaluation controls"),
            run_id,
            num_batches,
            max_examples,
            action,
        ]
    )
    return action, controls, max_examples, mo, num_batches, run_id


@app.cell
def _(action, max_examples, num_batches, run_id):
    from pathlib import Path as _Path

    import marimo as _mo
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.config.io import load_experiment_config
    from src.data.batching import collate_token_batches
    from src.data.dataset import TokenDataset
    from src.diffusion.corrupt import MaskedDiscreteCorruptionPolicy
    from src.diffusion.schedule import sample_timesteps
    from src.models.factory import build_model_from_experiment
    from src.store import DatasetStore as _DatasetStore
    from src.store import RunStore as _RunStore
    from src.tokenization.tokenizer import CharacterTokenizer, Tokenizer
    from src.training.checkpoint import load_checkpoint
    from src.utils.device import get_device

    batch_rows: list[dict[str, float | int | str | None]] = []
    confidence_rows: list[dict[str, float]] = []
    decoded_rows: list[dict[str, float | int | str]] = []
    example_metric_rows: list[dict[str, float | int]] = []
    position_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int | str | None]] = []
    status = _mo.md("Press `Evaluate` to score the selected run.")

    if action.value and run_id.value != "none":
        run = _RunStore().get(run_id.value)
        checkpoint = run.latest_checkpoint or run.best_checkpoint
        if checkpoint is None:
            status = _mo.md("No checkpoint found for the selected run.")
        else:
            config = load_experiment_config(run.config_path)
            artifact = _DatasetStore().get(run.dataset_id)
            config.model.vocab_size = artifact.vocab_size
            config.model.max_seq_len = artifact.seq_len
            config.dataset.seq_len = artifact.seq_len
            config.diffusion.mask_token_id = artifact.mask_token_id

            tokenizer = Tokenizer.from_file(
                artifact.stats_path.parent / "tokenizer.json"
            )
            raw_tokenizer = tokenizer.load()
            special_tokens = {
                artifact.pad_token_id: "<pad>",
                artifact.mask_token_id: "<mask>",
            }
            if isinstance(raw_tokenizer, CharacterTokenizer):
                special_tokens[raw_tokenizer.unk_token_id] = "<unk>"

            def render_token(token_id: int) -> str:
                if token_id in special_tokens:
                    return special_tokens[token_id]
                if isinstance(raw_tokenizer, CharacterTokenizer):
                    return raw_tokenizer.itos.get(token_id, "<unk>")
                token = raw_tokenizer.id_to_token(token_id)
                if token is not None:
                    return token.replace("Ġ", " ")
                return tokenizer.decode([token_id])

            def render_sequence(token_ids: list[int], limit: int = 96) -> str:
                rendered = "".join(render_token(token_id) for token_id in token_ids)
                return rendered[:limit] + ("..." if len(rendered) > limit else "")

            model = build_model_from_experiment(config)
            checkpoint_data = load_checkpoint(checkpoint, get_device())
            model.load_state_dict(checkpoint_data["model_state_dict"])
            model = model.to(get_device())
            model.eval()

            loader = DataLoader(
                TokenDataset(artifact.val_path),
                batch_size=config.training.batch_size,
                shuffle=False,
                collate_fn=collate_token_batches,
            )
            policy = MaskedDiscreteCorruptionPolicy(
                num_steps=config.diffusion.num_steps,
                schedule_name=config.diffusion.schedule_name,
                mask_token_id=config.diffusion.mask_token_id,
            )

            num_steps = config.diffusion.num_steps
            seq_len = artifact.seq_len
            timestep_counts = torch.zeros(num_steps, dtype=torch.long)
            position_mask_counts = torch.zeros(seq_len, dtype=torch.long)
            position_correct_counts = torch.zeros(seq_len, dtype=torch.long)
            position_loss_sums = torch.zeros(seq_len, dtype=torch.float)
            total_masked_tokens = 0
            total_masked_correct = 0
            total_loss_sum = 0.0
            total_sequences = 0
            decoded_limit = int(max_examples.value)

            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    if batch_index >= int(num_batches.value):
                        break

                    batch = batch.to(get_device())
                    timesteps = sample_timesteps(
                        batch.shape[0],
                        config.diffusion.num_steps,
                        device=get_device(),
                    )
                    corrupted, mask = policy.corrupt(batch, timesteps)
                    logits = model(corrupted, timesteps)
                    log_probs = F.log_softmax(logits, dim=-1)
                    predictions = logits.argmax(dim=-1)
                    prediction_confidence = log_probs.exp().amax(dim=-1)
                    token_nll = -log_probs.gather(
                        dim=-1,
                        index=batch.unsqueeze(-1),
                    ).squeeze(-1)

                    masked_tokens = int(mask.sum().item())
                    masked_correct = int(((predictions == batch) & mask).sum().item())
                    batch_loss_sum = float(token_nll[mask].sum().item())
                    batch_loss = batch_loss_sum / max(1, masked_tokens)
                    batch_accuracy = masked_correct / max(1, masked_tokens)

                    total_sequences += int(batch.shape[0])
                    total_masked_tokens += masked_tokens
                    total_masked_correct += masked_correct
                    total_loss_sum += batch_loss_sum
                    timestep_counts += torch.bincount(
                        timesteps.detach().cpu(),
                        minlength=num_steps,
                    )
                    position_mask_counts += mask.sum(dim=0).detach().cpu()
                    position_correct_counts += (
                        ((predictions == batch) & mask).sum(dim=0).detach().cpu()
                    )
                    position_loss_sums += (
                        token_nll.masked_fill(~mask, 0.0).sum(dim=0).detach().cpu()
                    )

                    batch_rows.append(
                        {
                            "batch_index": batch_index,
                            "masked_loss": round(batch_loss, 4),
                            "masked_accuracy": round(batch_accuracy, 4),
                            "masked_fraction": round(
                                masked_tokens / max(1, batch.numel()),
                                4,
                            ),
                            "mean_timestep": round(
                                float(timesteps.float().mean().item()),
                                2,
                            ),
                            "masked_tokens": masked_tokens,
                        }
                    )

                    masked_confidences = prediction_confidence[mask].detach().cpu()
                    confidence_rows.extend(
                        {"confidence": float(value)}
                        for value in masked_confidences.tolist()
                    )

                    for example_index in range(batch.shape[0]):
                        example_mask = mask[example_index]
                        example_masked_tokens = int(example_mask.sum().item())
                        example_correct = int(
                            (
                                (predictions[example_index] == batch[example_index])
                                & example_mask
                            )
                            .sum()
                            .item()
                        )
                        example_loss = float(
                            token_nll[example_index][example_mask].mean().item()
                        )
                        example_metric_rows.append(
                            {
                                "batch_index": batch_index,
                                "example_index": example_index,
                                "timestep": int(timesteps[example_index].item()),
                                "masked_tokens": example_masked_tokens,
                                "masked_fraction": example_masked_tokens / seq_len,
                                "masked_loss": example_loss,
                                "masked_accuracy": example_correct
                                / max(1, example_masked_tokens),
                            }
                        )

                        if len(decoded_rows) >= decoded_limit:
                            continue

                        decoded_rows.append(
                            {
                                "batch_index": batch_index,
                                "example_index": example_index,
                                "timestep": int(timesteps[example_index].item()),
                                "masked_tokens": example_masked_tokens,
                                "masked_loss": round(example_loss, 4),
                                "masked_accuracy": round(
                                    example_correct / max(1, example_masked_tokens),
                                    4,
                                ),
                                "target": render_sequence(
                                    batch[example_index].detach().cpu().tolist()
                                ),
                                "corrupted": render_sequence(
                                    corrupted[example_index].detach().cpu().tolist()
                                ),
                                "prediction": render_sequence(
                                    predictions[example_index].detach().cpu().tolist()
                                ),
                            }
                        )

            mean_loss = total_loss_sum / max(1, total_masked_tokens)
            masked_reconstruction_ppl = None
            if mean_loss < 20:
                masked_reconstruction_ppl = round(
                    float(torch.exp(torch.tensor(mean_loss))),
                    4,
                )

            summary_rows.append(
                {
                    "run_id": run.run_id,
                    "checkpoint": str(_Path(checkpoint).name),
                    "eval_batches": min(len(batch_rows), int(num_batches.value)),
                    "evaluated_sequences": total_sequences,
                    "masked_tokens": total_masked_tokens,
                    "masked_loss": round(mean_loss, 4),
                    "masked_reconstruction_ppl": masked_reconstruction_ppl,
                    "masked_accuracy": round(
                        total_masked_correct / max(1, total_masked_tokens),
                        4,
                    ),
                }
            )

            for position in range(seq_len):
                masked_count = int(position_mask_counts[position].item())
                position_rows.append(
                    {
                        "position": position,
                        "masked_rate": masked_count / max(1, total_sequences),
                        "masked_accuracy": float(
                            position_correct_counts[position].item()
                        )
                        / max(1, masked_count),
                        "masked_loss": float(position_loss_sums[position].item())
                        / max(1, masked_count),
                    }
                )

            status = _mo.md(
                "Diagnostics are computed on freshly resampled validation corruption, "
                "so rerunning changes the exact curves and examples."
            )
    return (
        batch_rows,
        confidence_rows,
        decoded_rows,
        example_metric_rows,
        position_rows,
        status,
        summary_rows,
    )


@app.cell
def _(
    batch_rows: list[dict[str, float | int | str | None]],
    mo,
    status,
    summary_rows: list[dict[str, float | int | str | None]],
):
    summary = (
        mo.md("No evaluation metrics yet.")
        if not summary_rows
        else mo.ui.table(summary_rows)
    )
    batches = (
        mo.md("Per-batch diagnostics will appear after evaluation.")
        if not batch_rows
        else mo.ui.table(batch_rows)
    )
    summary_view = mo.vstack([mo.md("## Evaluation summary"), status, summary, batches])
    return (summary_view,)


@app.cell
def _(
    confidence_rows: list[dict[str, float]],
    example_metric_rows: list[dict[str, float | int]],
    mo,
    position_rows: list[dict[str, float | int]],
):
    import altair as alt
    import polars as pl

    if not example_metric_rows:
        chart_output = mo.md("Run an evaluation to render charts.")
    else:
        example_df = pl.DataFrame(example_metric_rows)
        position_df = pl.DataFrame(position_rows)
        confidence_df = pl.DataFrame(confidence_rows)

        timestep_chart = (
            alt.Chart(example_df)
            .mark_circle(
                size=48,
                opacity=0.45,
            )
            .encode(
                x=alt.X("timestep:Q", title="Timestep"),
                y=alt.Y("masked_loss:Q", title="Per-sequence masked loss"),
                color=alt.Color("masked_accuracy:Q", title="Masked accuracy"),
                tooltip=[
                    alt.Tooltip("batch_index:Q", title="Batch"),
                    alt.Tooltip("example_index:Q", title="Example"),
                    alt.Tooltip("masked_tokens:Q", title="Masked tokens"),
                    alt.Tooltip("masked_loss:Q", title="Masked loss", format=".4f"),
                    alt.Tooltip("masked_accuracy:Q", title="Accuracy", format=".3f"),
                ],
            )
            .properties(title="Loss vs. timestep", height=260)
        )

        confidence_chart = (
            alt.Chart(confidence_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "confidence:Q",
                    bin=alt.Bin(maxbins=24),
                    title="Masked-token prediction confidence",
                ),
                y=alt.Y("count():Q", title="Token count"),
                tooltip=[alt.Tooltip("count():Q", title="Count")],
            )
            .properties(title="Confidence histogram", height=260)
        )

        position_loss_chart = (
            alt.Chart(position_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("position:Q", title="Token position"),
                y=alt.Y("masked_loss:Q", title="Average masked loss"),
                tooltip=[
                    alt.Tooltip("position:Q", title="Position"),
                    alt.Tooltip("masked_loss:Q", title="Masked loss", format=".4f"),
                    alt.Tooltip("masked_accuracy:Q", title="Accuracy", format=".3f"),
                    alt.Tooltip("masked_rate:Q", title="Mask rate", format=".3f"),
                ],
            )
            .properties(title="Position-wise loss", height=260)
        )

        position_accuracy_chart = (
            alt.Chart(position_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("position:Q", title="Token position"),
                y=alt.Y("masked_accuracy:Q", title="Masked accuracy"),
                color=alt.value("#d97706"),
                tooltip=[
                    alt.Tooltip("position:Q", title="Position"),
                    alt.Tooltip("masked_accuracy:Q", title="Accuracy", format=".3f"),
                    alt.Tooltip("masked_rate:Q", title="Mask rate", format=".3f"),
                ],
            )
            .properties(title="Position-wise accuracy", height=260)
        )

        chart_output = mo.vstack(
            [
                mo.ui.altair_chart(timestep_chart | confidence_chart),
                mo.ui.altair_chart(position_loss_chart | position_accuracy_chart),
            ]
        )
    return (chart_output,)


@app.cell
def _(decoded_rows: list[dict[str, float | int | str]], mo):
    decoded_view = (
        mo.md("Decoded examples will appear after evaluation.")
        if not decoded_rows
        else mo.vstack(
            [
                mo.md("## Decoded examples"),
                mo.ui.table(decoded_rows),
            ]
        )
    )
    return (decoded_view,)


@app.cell
def _(chart_output, controls, decoded_view, mo, summary_view):
    mo.vstack(
        [
            mo.md("# Eval Lab"),
            controls,
            summary_view,
            chart_output,
            decoded_view,
        ],
        gap=1.25,
    )
    return


if __name__ == "__main__":
    app.run()
