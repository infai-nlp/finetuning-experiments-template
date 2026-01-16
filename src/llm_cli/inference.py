from pathlib import Path

import click
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .utils import save_dict_to_json


def apply_chat_template(prompts, tokenizer):
    formatted_prompts = []
    for prompt in prompts:
        if isinstance(prompt, list):  # sys prompt
            messages = [
                {"role": "system", "content": prompt[0]},
                {"role": "user", "content": prompt[1]},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # for Qwen3, etc.
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts


def run_inference(config: dict, adapter: bool):
    llm = LLM(
        model=config["model"],
        trust_remote_code=True,
        enable_lora=True if adapter else False,
        max_lora_rank=256,
        enforce_eager=True,
        **config["infer"]["llm_params"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)

    ds = load_from_disk(config["dataset"])
    test_ds = ds["test"] #.shuffle(seed=42).select(range(100))  # assumes a DatasetDict with a "test" split

    ids = [ex["id"] for ex in test_ds]
    prompts = [ex["prompt"] for ex in test_ds]
    references = [ex["completion"] for ex in test_ds]

    if config["infer"]["apply_chat_template"]:
        prompts = apply_chat_template(prompts, tokenizer)

    sp = SamplingParams(**config["infer"]["sampling_params"])

    lora_request = LoRARequest("lora", 1, config["adapter_dir"]) if adapter else None

    if adapter:
        click.secho(f"\nUsing lora adapter {config['adapter_dir']}", fg="green")

    results = llm.generate(prompts, sampling_params=sp, lora_request=lora_request)

    rows = []
    for idx, out, prompt, reference in zip(
        ids, results, prompts, references, strict=True
    ):
        predictions = [g.text for g in out.outputs]
        rows.append(
            {
                "id": idx,
                "prompt": prompt,  # str
                "reference": reference,  # str  (ground-truth completion)
                "predictions": predictions,  # List[str] – the N sampled generations
            }
        )

    output_dir = Path(config["predictions_dir"])
    file = output_dir / "lora_predictions.json" if adapter else output_dir / "base_predictions.json"

    save_dict_to_json(rows, file)