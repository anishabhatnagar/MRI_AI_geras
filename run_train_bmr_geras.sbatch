#!/bin/bash
#SBATCH --job-name=mri_ai_bmr_geras
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=radiology
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/scratch/rt2741/breast_fm/mri_ai_bmr_geras_%j.out
#SBATCH --error=/gpfs/scratch/rt2741/breast_fm/mri_ai_bmr_geras_%j.err

set -euo pipefail

cd /gpfs/scratch/rt2741/breast_fm

source /gpfs/scratch/rt2741/miniconda3/etc/profile.d/conda.sh
conda activate bmr

# Trains the original Geras-lab MRI_AI model (Witowski et al. 2022) on the
# BMR dataset, using the pickles produced by
# dataset_creation_scripts/run_convert_bmr_to_mri_ai.sbatch.
#
# --half False avoids needing an apex build (see MRI_AI/train.py's now-lazy
# apex import). Once a working apex build is confirmed on this cluster's
# CUDA toolkit, switch to --half True for mixed precision.
#
# --use_neptune defaults True: fill in your real wandb project/entity in
# initialize_wandb() (train.py:41-43) and `wandb login` first, or pass
# --use_neptune False to skip wandb entirely.

DATA_ROOT=/gpfs/scratch/rt2741/mri_ai_bmr_data

python train.py \
  --metadata       "${DATA_ROOT}/metadata.pkl" \
  --datalist       "${DATA_ROOT}/datalist.pkl" \
  --subgroup_data  "${DATA_ROOT}/subgroup_data.pkl" \
  --input_type sub_t1c2 \
  --architecture 3d_resnet18 \
  --label_type cancer \
  --batch_size 2 \
  --num_epochs 75 \
  --lr 2e-5 \
  --half False \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --gpus 1 \
  --ddp False \
  --logdir "${DATA_ROOT}/logs" \
  --experiment bmr_geras_baseline

# Smoke test first (edit num_epochs/sample counts down), e.g.:
#   --num_epochs 1 --number_of_training_samples 4 --number_of_validation_samples 4
