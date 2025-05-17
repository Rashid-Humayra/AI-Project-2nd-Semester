#!/bin/bash
#SBATCH --job-name=FirstTrain
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100-20:1
#SBATCH --time=2-00:00:00
#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out


python scripts/run.py --config-name=wanjinyou_big \
dataset_name=waymo_multi_view case_name=segment-11379226583756500423_6230_810_6250_810_with_camera_labels_old \
exp_name=exp_coeff_0.15 dataset.shutter_coefficient=0.15 mode=train_hdr_shutter +work_dir=$(pwd) 