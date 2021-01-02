#!/bin/bash
DIV=$1
BETA=$2
W=$3
RUN=$4
TARGET_LABELS=$5

if [[ "${DIV}" == "w_beta" ]]; then
    D_GRAD_PENALTY="10.0"
else
    D_GRAD_PENALTY="0.0"
fi

set -x
python -u -B train_da_relax.py \
    --exp_name="usps_mnist" \
    --log_sub_dir="tlabels-${TARGET_LABELS}/w-${W}_div-${DIV}_beta-${BETA}/run-${RUN}" \
    --refresh_log_dir="True" \
    --config_file="usps_mnist_config" \
    --args="target_labels='${TARGET_LABELS}'" \
    --args="d_loss_name='${DIV}'" \
    --args="d_relax=${BETA}" \
    --args="d_grad_penalty=${D_GRAD_PENALTY}" \
    --args="d_loss_w=${W}"