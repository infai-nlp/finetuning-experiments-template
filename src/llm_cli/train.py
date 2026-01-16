import os
from pathlib import Path

import click
import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from .utils import save_dict_to_json


def print_train_info(trainer: SFTTrainer):
    click.secho(100 * "-", fg="green")
    click.secho("Trainable params:", fg="green")
    
    trainer.model.print_trainable_parameters()

    click.secho(
        f"Attn {trainer.model.config._attn_implementation}, {torch.backends.cuda.matmul.allow_tf32}",
        fg="green",
    )
    print(f"First train sample\n {trainer.train_dataset[0]}")
    print(f"Last train sample\n {trainer.train_dataset[-1]}")
    click.secho(f"SFTConfig: {trainer.args}", fg="green")
    click.secho(
        f"logging, evaluation, checkpointing @ step {trainer.args.logging_steps} {trainer.args.eval_steps} {trainer.args.save_steps}",
        fg="green",
    )
    click.secho(100 * "-", fg="green")


def format_conv(ds):
    """Wrap plain prompt/completion pairs into chat-style schema."""

    def wrap(ex):
        if isinstance(ex["prompt"], list):  # already multi-turn
            prompt = [
                {"role": "system", "content": ex["prompt"][0]},
                {"role": "user", "content": ex["prompt"][1]},
            ]
        else:
            prompt = [{"role": "user", "content": ex["prompt"]}]
        completion = [{"role": "assistant", "content": ex["completion"]}]
        return {"prompt": prompt, "completion": completion}

    return ds.map(wrap)


def load_dataset(path: Path, split: str = "train", conv: bool = False):
    """
    Load dataset from disk and optionally convert to chat-format.
    """
    click.secho(f"Loading dataset from {path} (split={split}, conv={conv})", fg="green")
    raw = load_from_disk(str(path))
    ds = raw[split] if isinstance(raw, DatasetDict) else raw
    return format_conv(ds) if conv else ds


def build_trainer(cfg: dict, local_rank: int) -> SFTTrainer:
    dataset_path = cfg.get("dataset", None)
    conv = True if cfg.get("model_type", None) == "chat" else False
    train_ds = load_dataset(dataset_path, split="train", conv=conv)
    val_ds = load_dataset(dataset_path, split="validation", conv=conv)

    #train_ds = train_ds.select(range(1024)) # for quick testing
    #val_ds = val_ds.select(range(32)) # for quick testing

    click.secho(f"train_ds size: {len(train_ds)}", fg="green")
    click.secho(f"val_ds size: {len(val_ds)}", fg="green")

    quantization_conf = cfg["train"].get("quantization")

    model_init_kwargs = dict(
        quantization_config=BitsAndBytesConfig(**quantization_conf) if quantization_conf else None,
        #attn_implementation="flash_attention_2",
        attn_implementation="sdpa", #native
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    sft_config = SFTConfig(
        **cfg["train"]["sft"],
        output_dir=cfg["adapter_dir"],
        local_rank=local_rank,
    )

    lora_config = LoraConfig(
        **cfg["train"]["lora"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        # lora_modules_to_save=["embed_tokens", "lm_head"],  # might be necessary for domain adaption, but needs merging otherwise not working with vllm
    )

    return SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(cfg["model"], **model_init_kwargs),
        processing_class=AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True),
        args=sft_config,
        peft_config=lora_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )


def run_training(cfg) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # should be 0,1 etc for num gpus
    click.secho(f"Using local_rank: {local_rank}", fg="green")

    trainer = build_trainer(cfg, local_rank)
    print_train_info(trainer)

    trainer.train()
    trainer.save_model(output_dir=cfg["adapter_dir"])

    click.secho(f"[train] ✓ LoRA adapter saved to {cfg['adapter_dir']}", fg="green")

    save_dict_to_json(cfg, Path(cfg["results_dir"]) / "config.json")
