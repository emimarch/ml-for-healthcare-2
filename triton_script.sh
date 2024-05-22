#!/bin/bash
#SBATCH --partition=gpu-v100-32g 
#SBATCH --time=08:00:00
#SBATCH --mem=9000MB
#SBATCH --output=first_try.log
#SBATCH --error=my_job.err
#SBATCH --gres=gpu:1



module load mamba
source activate codes2

nvidia-smi


python -u codes/train_causal_lm.py --per_device_train_batch_size 8 --block_size 800 --seed 42 --pretrained_model_name_or_path seeklhy/codes-1b --epochs 4 --lr 5e-6 --warmup_ratio 0.05 --checkpointing_steps 100000 --tensorboard_log_dir ./train_logs/codes-1b-model-1 --mode sft --output_ckpt_dir ./ckpts/codes-1b-model-1 --text2sql_data_dir ./data/train_reformatted.json --table_num 19 --column_num 76 > test2.out 
