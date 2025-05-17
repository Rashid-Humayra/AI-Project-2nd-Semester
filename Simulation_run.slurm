#!/bin/bash
#SBATCH --job-name=Simulation
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100-10:1
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out


export OPENAI_API_KEY=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

# if you train nerf
#python main.py -y config/waymo-1137.yaml -p "Add a police car in front of me, driving away fast." -s Simulation2

# python main.py -y config/waymo-1137.yaml -p "Add a porsche in front of me, driving away fast. The view should be moved 1 meters ahead" -s Simulation3


# python main.py -y config/waymo-1137.yaml -p "Move the added car to the left lane." -s Simulation4
# python main.py -y config/waymo-1137.yaml -p "Add a police car behind that chasing it." -s Simulation5
# python main.py -y config/waymo-1137.yaml -p "Delete other vehicles from the scene." -s Simulation6

python main.py -y config/waymo-1137.yaml -p "Add a porsche in front of me, driving away fast." -s styletransferedsimulation