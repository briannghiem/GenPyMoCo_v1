#!/bin/bash -l
 
export SUBMIT_DIR=/home/nghiemb/GenPyMoCo
export OUT_DIR=/home/nghiemb/GenPyMoCo/data

python ${SUBMIT_DIR}/gen_datasets.py > ${OUT_DIR}/log_gen_datasets.txt & disown
