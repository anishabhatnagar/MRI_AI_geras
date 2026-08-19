#!/bin/bash
#SBATCH --job-name=mri_ai_bmr_geras
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=19
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=radiology
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=anisha.bhatnagar@nyulangone.org
#SBATCH --output=/gpfs/scratch/ab10945/Breast_MR_WM/MRI_AI_geras/slurm/slurm_%j_%x.out
#SBATCH --error=/gpfs/scratch/ab10945/Breast_MR_WM/MRI_AI_geras/slurm/slurm_%j_%x.err
#SBATCH --open-mode=append

set -euo pipefail

cd /gpfs/scratch/ab10945/Breast_MR_WM/MRI_AI_geras

source /gpfs/scratch/ab10945/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/scratch/ab10945/RADFACT/envs/fmm_p312

# Trains the original Geras-lab MRI_AI model (Witowski et al. 2022) on the
# BMR dataset, using the pickles produced by
# dataset_creation_scripts/run_convert_bmr_to_mri_ai.sbatch.
#
# --architecture multi_channel --input_type three_channel matches the model
# the paper actually reports results for: pre/post1/post2 stacked as 3
# input channels. The paper explicitly found subtraction images (sub_t1c1/
# sub_t1c2) did NOT outperform this (Methods, "Subtraction images"), so that
# config is left out here (still available as an ablation later if wanted).
# With three_channel, dataset.py loads pre, post1, AND post2 -- unlike
# sub_t1c2, none of the converter's 3 shim files go unused.
#
# Not yet wired in: the paper's backbone was pre-trained on Kinetics-400
# (--weights <path> --weights_policy kinetics), which they found improved
# performance over training from scratch. Add that once Kinetics-pretrained
# 3D-ResNet18 weights are sourced.
#
#
# --half False avoids needing an apex build (see MRI_AI/train.py's now-lazy
# apex import). Once a working apex build is confirmed on this cluster's
# CUDA toolkit, switch to --half True for mixed precision.
#
# --use_neptune defaults True: fill in your real wandb project/entity in
# initialize_wandb() (train.py:41-43) and `wandb login` first, or pass
# --use_neptune False to skip wandb entirely.

DATA_ROOT=/gpfs/scratch/ab10945/mri_ai_bmr_data   
LOGS_ROOT=/gpfs/scratch/ab10945/Breast_MR_WM/MRI_AI_geras

python train.py \
  --metadata       "${DATA_ROOT}/metadata.pkl" \
  --datalist       "${DATA_ROOT}/datalist.pkl" \
  --subgroup_data  "${DATA_ROOT}/subgroup_data.pkl" \
  --input_type three_channel \
  --architecture multi_channel \
  --label_type cancer \
  --batch_size 2 \
  --num_epochs 75 \
  --lr 2e-5 \
  --half False \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --gpus 2 \
  --ddp False \
  --logdir "${LOGS_ROOT}/logs" \
  --experiment bmr_geras_baseline

# Smoke test first (edit num_epochs/sample counts down), e.g.:
#   --num_epochs 1 --number_of_training_samples 4 --number_of_validation_samples 4
