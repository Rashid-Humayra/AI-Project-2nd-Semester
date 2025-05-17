#!/bin/bash
#SBATCH --job-name=Evaluation
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-20:1
#SBATCH --time=1-00:00:00
#SBATCH --error=logs_eval/%j.out
#SBATCH --output=logs_eval/%j.out


python scripts/eval_mine.py \
  --base_data_dir /home/mundus/hrashid173/ChatSim/chatsim/background/mcnerf/ \
  --scenes data/waymo_multi_view/segment-11379226583756500423_6230_810_6250_810_with_camera_labels \
  --methods exp/segment-11379226583756500423_6230_810_6250_810_with_camera_labels/exp_coeff_0.15/test_images