#!/usr/bin/env bash
# #!/bin/bash
nvidia-smi


python run_lacrm_inf.py --use_pretrained True --model_name unet --LACRM True --rank unc --soft none --mix_beta 0.5 --beta 0.5 --begin_epoch 10 --aug True