from pathlib import Path
import click

from .metrics import compute_exact_match
from .utils import load_data_flat, write_results_section


def run_eval(config: dict, adapter: bool):
    """Evaluate predictions using specified metrics depending on adapter mode."""

    metrics = config["eval"]["our_task"]["metrics"]

    predictions_dir: Path = Path(config["predictions_dir"])
    result_dir: Path = Path(config["results_dir"])
    result_file = result_dir / "evaluation_results.json"

    top_key = "lora_predictions" if adapter else "base_predictions"
    predictions_file = predictions_dir / f"{top_key}.json"

    if not predictions_file.exists():
        click.secho(f"[our_task] Missing predictions file: {predictions_file}", fg="red")
        return

    results = {}
    ids, prompts, references, predictions = load_data_flat(predictions_file)

    click.secho(f"\nRunning evaluation of {predictions_file}\n", fg="green")
    for metric_name in metrics:
        if metric_name == "exact_match":
            metric_results = compute_exact_match(ids=ids, predictions=predictions, references=references)
        else:
            click.secho(f"Warning: Unknown metric '{metric_name}', skipping", fg="yellow")
            continue
        results[metric_name] = metric_results
        click.secho(f"{metric_name} results: {metric_results}", fg="green")

    write_results_section(
        path=result_file, top_key=predictions_file.stem, section_key="our_task", metrics=results
    )
