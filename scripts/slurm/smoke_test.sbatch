#!/bin/bash
#SBATCH --job-name=camtrolpp_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Full smoke test on one GPU. Edit the conda lines and E1 paths for another cluster.
#   1. CPU unit tests
#   2. equivalence with the paper's orbit results (needs the E1 outputs)
#   3. every camera action on one example image, sanity checks, evaluation
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate mview
PY=${PYTEST_PYTHON:-python}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

$PY -m pytest -q -p no:cacheprovider -W ignore tests/

E1=${E1_ROOT:-$HOME/camtrolpp_results/matched_step_ablation}
python -u tests/check_equivalence.py --e1 "$E1/outputs/matched_step" --e1_code_dir "$E1" --out outputs/smoke/equivalence

python -u run.py --image assets/examples/image000.png --config configs/*.yaml --out outputs/smoke/cameras
python -u tests/check_outputs.py outputs/smoke/cameras/image000
python -u evaluate.py outputs/smoke/cameras
echo "SMOKE_DONE end=$(date)"
