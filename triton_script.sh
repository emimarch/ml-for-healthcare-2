
#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=500M
#SBATCH --output=pi-gpu.out
#SBATCH --gres=gpu:1

module load gcc/8.4.0 cuda

module load mamba
mamba env create --file environment.yml
source activate codes

accelerate config --config_file accelerate_config.yaml

srun training.sh