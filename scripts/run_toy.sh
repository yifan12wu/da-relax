#!/bin/bash
DIV=js
BETA=0.0
RUN=0

if [ $DIV == w_beta ]
then
    D_GRAD_PENALTY=5.0
else
    D_GRAD_PENALTY=0.0
fi

python -u -B train_da_relax.py \
--exp_name=toy \
--log_sub_dir=div-${DIV}_beta-${BETA}/run-${RUN} \
--refresh_log_dir=True \
--config_file=toy_config \
--args="d_loss_name='${DIV}'" \
--args="d_relax=${BETA}" \
--args="d_grad_penalty=${D_GRAD_PENALTY}"