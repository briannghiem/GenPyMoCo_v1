#!/bin/bash -l
 
export SUBMIT_DIR=/home/nghiemb/GenPyMoCo
export OUT_DIR=/home/nghiemb/GenPyMoCo/cnn

python ${SUBMIT_DIR}/train.py > ${OUT_DIR}/log_train_nn.txt & disown
