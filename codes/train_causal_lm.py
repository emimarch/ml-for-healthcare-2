import argparse
import os
import math
import time
import random 
import numpy as np
import sys
import torch
from torch.cuda.amp import autocast, GradScaler
#torch.cuda.empty_cache()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_pt_dataset import PretrainDataset
from utils.load_sft_dataset import SFTSQLGenerationDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from datetime import datetime
'''
Training LLM using Huggingface Accelerate + Deepspeed.
'''

# Open a file for logging
#log_file = open("program_output.log", "a")

# Redirect stdout and stderr to the log file
#sys.stdout = log_file
#sys.stderr = log_file


def parse_option():
    parser = argparse.ArgumentParser()
    
    # global args
    parser.add_argument('--per_device_train_batch_size', type = int, default = 4,
                        help = 'batch size per gpu device.')
    parser.add_argument('--block_size', type = int, default = 8192,
                        help = 'block size, i.e., the length of training sequences.')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pretrained_model_name_or_path', type = str, default = "bigcode/starcoder")
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--lr', type = float, default = 5e-5, help = "5e-5 for pre-training, 5e-6 for fine-tuning.")
    parser.add_argument('--warmup_ratio', type = float, default = 0.0, help = "ratio of total training steps used for a linear warmup from 0 to max lr.")
    parser.add_argument('--checkpointing_steps', type = int, default = 300)
    parser.add_argument('--tensorboard_log_dir', type = str, default = "./train_logs")
    parser.add_argument('--mode', type = str, default = "pt")
    parser.add_argument('--output_ckpt_dir', type = str, default = "./ckpts")
    parser.add_argument('--save_all_states', action = 'store_true', 
        help = "whether to save states of model, optimizer, and lr scheduler for resuming training, otherwise only model states are saved.")

    # args for pre-training
    parser.add_argument('--pt_data_dir', type = str, default = "./data/corpus.bin")
    parser.add_argument('--resume_from_checkpoint', type = str, default = None, 
                            help = "resuming pre-training from a checkpoint")
    parser.add_argument('--resume_tag', type = str, default = None)
    
    # args for supervised fine-tuning
    parser.add_argument('--text2sql_data_dir', type = str, default = "./data/sft_train_text2sql.json")
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)
    
    opt = parser.parse_args()

    return opt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_model_optimizer_scheduler(checkpoint_folder, model, last_global_step, lr_scheduler):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "last_global_step": last_global_step,
    }

    print("==> saving model and optimizer <==")
    model.save_checkpoint(checkpoint_folder, last_global_step, checkpoint_state_dict)

    print("==> saving lr scheduler <==")
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_folder, str(last_global_step), "scheduler.pt"))

    print(f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={last_global_step}")
    return

def resume_model_and_optimizer(model, load_dir, tag):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag = tag, load_optimizer_states = True)
    
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict

    return last_global_step


def checkpoint_model_new(model, tokenizer, output_ckpt_dir, last_global_step):    
    '''
    Utility fuction for only checkpointing the model dictionary (i.e., only model parameters)
    '''
    ckpt_path = os.path.join(output_ckpt_dir, "ckpt-{}".format(last_global_step))
    print("checkpointing model state dict at {}".format(ckpt_path))
    # TODO: currently, there is a small bug that saves a full checkpoint data for each shard when enable zero1 and 2. 
    # See https://github.com/microsoft/DeepSpeed/issues/3303. solution: waiting upgrade of accelerate and deepspeed
    model.save_pretrained(
        ckpt_path, 
    )

    tokenizer.save_pretrained(ckpt_path)
    
    return

def sanity_check(input, target, tokenizer):
    print("Start Sanity Check -------->")
    for t, m in zip(input[:-1], target[1:]):
        decoded = tokenizer.decode([t])
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<-------- End Sanity Check")

def train(opt):
    print('Hello!')
    custom_home = '/scratch/work/marchee3/ML_for_healthcare/huggingface'
    device = torch.device('cuda:0')
    print('Current device: {}'.format(torch.cuda.current_device()))
    set_seed(opt.seed)
    print('Here')

    total_batch_size = opt.per_device_train_batch_size


    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path, cache_dir = custom_home)
    print('Qui1')
    torch.cuda.empty_cache()
    print('Qui1.25')
    model = AutoModelForCausalLM.from_pretrained(opt.pretrained_model_name_or_path, cache_dir = custom_home)
    print('Qui1.5')
    model.to(device)
    print('Qui2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print('Qui3')
    model.config.pad_token_id = tokenizer.eos_token_id
    print('Qui4')
    
    # enable gradient checkpointing to save GPU memory, but this action would slowdown the training speed 20-30%
    model.gradient_checkpointing_enable()
    print('Qui5')


    if opt.mode == "sft":
        dataset = SFTSQLGenerationDataset(opt.text2sql_data_dir, tokenizer, opt.block_size, "train", opt.table_num, opt.column_num, None)
        print('Qui6')
    else:
        raise ValueError("opt.mode should be in [pt, sft].")
    dataloader = DataLoader(dataset, batch_size = opt.per_device_train_batch_size, shuffle = True, drop_last = True)
    print('Qui7')
    num_total_batches = math.ceil(opt.epochs * math.ceil(len(dataset) / total_batch_size)) # number of total batches
    optimizer = AdamW(model.parameters(), lr = opt.lr, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 0.1)
    print('QUi8')
    num_warm_up_batches = max(int(num_total_batches * opt.warmup_ratio), 1)
    print('Qui9')
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer = optimizer, 
        warmup_epochs = num_warm_up_batches,
        max_epochs = num_total_batches,
        warmup_start_lr = 0.0, 
        eta_min = 0.1 * opt.lr
    )


    print(type(optimizer))
    print(type(model))
    print(type(dataloader))
    print(type(lr_scheduler))

    accumulation_loss = 0
    global_completed_steps = 0
    model.train()
    print('QuiIDK')
    st = time.time()
    st_sr = time.strftime("%H:%M:%S")
    print('Start time:{}'.format(st_sr))

    # Start scaler
    scaler = GradScaler()

    for epoch in range(opt.epochs):
        print('----------------------------------------')
        print("Start training epoch:", epoch+1)
        print('----------------------------------------')
        for batch_idx, batch in enumerate(dataloader):
            bts = time.time()
            print('Batch: {}'.format(batch_idx))
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            print('Moved batch to device')

            # Added for mixed precision
            with autocast(dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
            
            
            print('Before loss backward')
            #loss.backward()
            #optimizer.step()

            # ADDED BECAUSE OF MIXED PRECISION
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            lr_scheduler.step()
            optimizer.zero_grad()

            accumulation_loss += loss.detach().float()
            bte = time.time()
            if batch_idx % 250 == 0:
                print('Accumulated loss: {}'.format(accumulation_loss))
                print('Batch loss: {}'.format(loss.detach().float()))
                print('End batch at time: {}'.format(time.strftime("%H:%M:%S")))
                print('Time taken to process time: {}'.format(bte-bts))
                print('Time since training start: {}'.format(bte-st))
                print('################################################')

        print("in the end of an epoch, save a checkpoint")
        checkpoint_model_new(model, tokenizer, opt.output_ckpt_dir, global_completed_steps)
        if opt.save_all_states:
            checkpoint_model_optimizer_scheduler(opt.output_ckpt_dir, model, global_completed_steps, lr_scheduler)


if __name__ == "__main__":
    opt = parse_option()
    train(opt)