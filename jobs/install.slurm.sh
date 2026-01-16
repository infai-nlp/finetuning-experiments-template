#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --job-name=install
#SBATCH --partition=????
#SBATCH --nodes=1
#SBATCH --mincpus=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=????


######### LOAD MODULES #########

module load release/24.10
module load GCCcore/13.3.0

module load Python/3.12.3
module load CUDA/12.6.0

######### Setting .env variables #########

VENV_DIR=""
echo "Setting VENV_DIR: $VENV_DIR"

SRC_DIR="$HOME"
echo "Setting SRC_DIR: $SRC_DIR"
cd "$SRC_DIR"

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

echo "TORCH_CUDA_ARCH_LIST"
echo $TORCH_CUDA_ARCH_LIST

# Set poetry virtual env path 
poetry config virtualenvs.path $VENV_DIR
poetry install --no-interaction

# *Exceptional* call of pip to use --no-build-isolation flag
poetry run pip install --no-build-isolation "flash-attn==2.8.0.post2"

######### VERIFY INSTALL #########

echo "Verifying installation.."
poetry run llm-cli --help

# Show what is installed in my environment for debug purposes
echo "Installed packages:"
poetry show

echo "Where is my env?"
poetry env info 
