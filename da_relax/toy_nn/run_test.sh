#!/bin/bash

python -u -B run.py \
    --args="dataset='cls'" \
    --args="n_train=30" \
    --args="total_train_steps=20000" \
    --args="n_conv_layers=1" \
    --args="n_fc_layers=0" \
    --args="loss='l2'" \
    --args="batch_size=0" \
    --args="optimizer_name='SGD'" \
    --args="init_lr=0.01" \
    --args="scheduled_lrs=((0.01, 1000),)" \
    --args="init_wd=0.001" \
    --args="scheduled_wds=((0.0, 50000),)" \
    --args="init_mul=0.0001"
