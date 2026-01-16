#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=exp
#SBATCH --partition=????
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=????

NUM_GPUS=1

if [ -z "$1" ]; then
  echo "Error: No config file provided."
  echo "Usage: sbatch $0 <config.yaml>"
  exit 1
fi

CONFIG="$1"

echo "Using config: $CONFIG"

######### LOAD MODULES #########

module load release/24.10
module load GCCcore/13.3.0

module load Python/3.12.3
module load CUDA/12.6.0

nvidia-smi -L

######### SRC DIR #########

SRC_DIR="$HOME"
echo "SRC_DIR: $SRC_DIR"
cd "$SRC_DIR"

######### LOAD .ENV #########

ENV_FILE="$SRC_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  echo "Loading $ENV_FILE"
  set -o allexport
  . "$ENV_FILE"
  set +o allexport
else
  echo "No .env found at $ENV_FILE"
  exit 1
fi

######### CHECK VIRTUAL ENVIRONMENT AND SPAWN SHELL #############

poetry env info 
source $(poetry env info --path)/bin/activate
# source $VENV_DIR/bin/activate

###########################################################################
### Experiment (adapter training, inference on test, evaluation on test ###
###########################################################################

hf auth login --token "$HF_HUB_TOKEN"

######### RUN TRAIN -> /base_dir/model_id/experiment_id/results/adapters/ #########
echo "Starting training.."
poetry run llm-cli train --config $CONFIG

######### RUN INFERENCE -> /base_dir/model_id/experiment_id/results/predictions #########
echo "Starting inference for adapter model.."
poetry run llm-cli inference --config $CONFIG --adapter
echo "Starting inference for baseline model.."
poetry run llm-cli inference --config $CONFIG --no-adapter

######### RUN EVAL -> /base_dir/model_id/experiment_id/results/evaluation_results.json #########
echo "Starting evaluation .."
poetry run llm-cli evaluation --config $CONFIG --keep-going --verbose
