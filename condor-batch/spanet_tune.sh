#!/bin/bash

date
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -u 127590
python3 -m spanet.tune /eos/user/j/jmitic/hhh/options_files/hhh_v33_SVB.json --gpus 4
date
exit
