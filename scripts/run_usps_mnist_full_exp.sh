#!/bin/bash

N_RUNS=5

# run without adaptation
for ((i=0;i<N_RUNS;i++)); do
    ./scripts/run_toy.sh js 0.0 0.0 $i
done

# run dann without relaxation
for ((i=0;i<N_RUNS;i++)); do
    ./scripts/run_toy.sh js 0.0 1.0 $i
done

# run dann with relaxation

DIVS="js_beta js_sort"
BETAS="2.0 4.0"

for div in ${DIVS}; do
    for beta in ${BETAS}; do
        for ((i=0;i<N_RUNS;i++)); do
            ./scripts/run_toy.sh ${div} ${beta} 1.0 $i
        done
    done
done