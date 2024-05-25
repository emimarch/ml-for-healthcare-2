#!/bin/bash
#SBATCH --partition=gpu-v100-32g 
#SBATCH --time=22:00:00
#SBATCH --mem=10000MB
#SBATCH --output=valid_try.log
#SBATCH --error=valid.err
#SBATCH --gres=gpu:1



module load mamba
source activate codes2

nvidia-smi



python -u codes/test_causal_lm.py --llm_path ckpts/codes-1b-model-1/ckpt-0  --sic_path ./sic_ckpts/sic_bird --table_num 6 --column_num 10 --dataset_path data/valid_reformatted.json  --database_path data/database/mimic_iv_cxr/silver/mimic_iv_cxr.db --max_tokens 1200 --max_new_tokens 400 --dset valid > valid_log.log
