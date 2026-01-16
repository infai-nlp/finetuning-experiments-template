# finetuning-experiments-template

CLI application for LoRA Finetuning and Evaluating Open Source LLMs on Huggingface Hub. Every model should have its config file under `configs/` (See examples).

## Installation

First, install `poetry>=2.0.0` on your system.

For a local installation on your notebook execute `poetry install --all-groups`. However, note that the current implementation uses flash attention 2, which has to be installed and supported by your system. You can also change `attention_implementation` (train.py) to `sdpa`, which is natively supported.

You can activate the env with `eval $(poetry env activate)`, then run `llm-cli` commands, or just use `poetry run llm-cli train/inference/evaluation`.

Installation on the SLURM managed cluster is done with the installation script by

```bash
sbatch jobs/install.slurm
```

**Note:** Set the $VENV_DIR Variable before to your workspace directory with persistent storage.

**Note:** Poetry only manages Python wheels. It does not install: CUDA Toolkit, GPU drivers, cuDNN / NCCL, ...

## Dependency Management

We use **Poetry** for dependency and environment management.

> 🚫 Do **not** use `pip install` or create `requirements.txt`.  
> ✅ Instead, always use Poetry commands so dependencies stay consistent across environments.

To add a new package do:

```bash
poetry add <package-name>
```

Then commit the `pyproject.toml` and `poetry.lock` to git.

## Using the CLI

**Prerequisite:** Make sure to specifiy a HF datasets in the configuration yaml, with at least "prompt" (str) and "completion" (str) field.

The CLI contains three different command branches:

1. `train`: For lora finetuning an LLM.
    This command only requires a configuration yaml

    ```bash
    poetry run llm-cli train --config /path/to/config.yaml 
    ```

2. `inference`: Inference on the test set of dataset used for training.
    This command requires a configuration yaml and a --adapter/--no-adapter flag, to specify if you want to infer on the base or lora model.

    ```bash
    poetry run llm-cli inference --adapter --config /path/to/config.yaml 
    poetry run llm-cli inference --no-adapter --config /path/to/config.yaml 
    ```

3. `evaluation`: Evaluating LLMs on different tasks. Options:
    - `--task: bool` Evaluate on test split of the dataset used for training
    - `--adapter-mode: str` Options `no-adapter|adapter|both`. The evaluation for the specified tasks is carried out with base model and/or adapter model (*default: both*)
    - `--stop-on-error/--keep-going`: Stop on first error or continue with other evaluations.  [default: stop-on-error]
    - `--verbose`: Print full tracebacks on errors.

   ```bash
   poetry run llm-cli evaluation --config configs/my_model.yaml --task --adapter-mode both --keep-going --verbose
   ```

## Result directory

See: e.g., configs/test.yaml

```bash
$base_dir/
└── $model/
    └── $experiment_name/
        └── results/
            ├── config.json
            ├── evaluation_results.json
            ├── predictions/
            │   ├── base_predictions.json
            │   └── lora_predictions.json
            └── adapter/
                ├── adapter_config.json
                ├── adapter_model.bin
                ├── training_args.bin
                └── and so on    
```
