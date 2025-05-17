#!/bin/bash
#SBATCH --job-name=styletransfer
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100-10:1
#SBATCH --time=1-00:00:00
#SBATCH --error=logs_styletransfer/%j.out
#SBATCH --output=logs_styletransfer/%j.out

python overwrite_with_cyclegan.py
