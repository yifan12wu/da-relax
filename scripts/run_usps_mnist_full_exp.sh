#!/bin/bash

N_RUNS=1
TARGET_LABELS="01234 56789 0123456789"
RUN_SH="./scripts/run_usps_mnist.sh"
DIVS="js_beta js_sort"
BETAS="2.0 4.0"


for t_label in ${TARGET_LABELS}; do
    # run without adaptation
    for ((i=0;i<N_RUNS;i++)); do
        ${RUN_SH} js 0.0 0.0 ${i} ${t_label}
    done

    # run dann without relaxation
    for ((i=0;i<N_RUNS;i++)); do
        ${RUN_SH} js 0.0 1.0 ${i} ${t_label}
    done

    # run dann with relaxation
    for div in ${DIVS}; do
        for beta in ${BETAS}; do
            for ((i=0;i<N_RUNS;i++)); do
                ${RUN_SH} ${div} ${beta} 1.0 ${i} ${t_label}
            done
        done
    done

done