import json
import re
from pathlib import Path
from typing import Any, Dict

import click


def load_data_flat(path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load JSON file with evaluation data"""
    with open(path, "r") as f:
        data = json.load(f)

    ids, prompts, references, predictions = [], [], [], []

    for item in data:
        for i, pred in enumerate(item["predictions"]):
            id = item["id"] if type(item["id"]) == str else str(item["id"])
            ids.append(f"{i}_" + id)
            prompts.append(item["prompt"])
            references.append(item["reference"])
            predictions.append(pred)

    return ids, prompts, references, predictions


def slugify(model_name: str) -> str:
    slug_rx = re.compile(r"[^A-Za-z0-9._-]+")  # keep alnum, dot, dash, underscore
    return slug_rx.sub("_", model_name).strip("_")


def prepare_paths(cfg):
    """
    Prepare and attach result-related directories to the config dict.

    Example:
    ./experiments/
    └── meta-llama_Llama-3.2-3B/
        └── no-deepspeed-lower_lr/
            └── results/
                ├── predictions/
                ├── adapter/
                └── metric_results.json
    """
    base_dir = Path(cfg.get("base_dir", "./experiments"))
    model_name = slugify(cfg.get("model"))
    exp_name = cfg.get("experiment_name", "unnamed_experiment")

    results_dir = (base_dir / model_name / exp_name / "results").resolve()
    adapter_dir = (results_dir / "adapter").resolve()
    predictions_dir = (results_dir / "predictions").resolve()

    for d in [results_dir, adapter_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg["results_dir"] = str(results_dir)
    cfg["adapter_dir"] = str(adapter_dir)
    cfg["predictions_dir"] = str(predictions_dir)

    return cfg


def write_results_section(
    path: Path,
    top_key: str,  # "lora_predictions" or "base_predictions"
    section_key: str,  # e.g. "rouge", "bertscore", "geval"
    metrics: Dict[str, Any],  # payload to store under that section
) -> None:
    """
    Writes/updates evaluation results like:
    {
      top_key: {
        section_key: { ...metrics... }
      }
    }
    """
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    bucket = data.setdefault(top_key, {})
    bucket.setdefault(section_key, {})
    bucket[section_key].update(metrics)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))

    click.secho(f"Metric '{section_key}' results written to {path} for {top_key}", fg="green")


def save_dict_to_json(data: dict, file_path: Path):
    """
    Save a dictionary to a JSON file, printing success or error messages.
    """
    file_path = Path(file_path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        click.secho(f"Successfully saved {file_path}", fg="green")

    except Exception as e:
        click.secho(f"Failed to save {file_path}: {e}", fg="red")
