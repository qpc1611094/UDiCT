#!/usr/bin/env bash
nvidia-smi


python run_lacrm.py --model_name unet --rank unc --soft none --mix_beta 0.5 --lr 3e-4 --data 0.1 --output_type sigmoid --u_thre 0.5 --beta 0.5 --begin_epoch 10 --ram_beta down --batchsize 4 --EPOCH 400 --devices 6,7

