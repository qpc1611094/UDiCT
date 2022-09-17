#!/usr/bin/env bash
nvidia-smi

python run_lacrm.py --model_name unet --rank unc --soft mse --mix_beta 0.5 --lr 1e-4 --datanum 4000 --output_type sigmoid --u_thre 0.5 --beta 1.0 --begin_epoch 10 --ram_beta none --batchsize 32 --LACRM True --epoch 400 --devices 6,7 --if_scale True