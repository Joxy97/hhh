#!/bin/bash

date
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m spanet.train -of /eos/user/j/jmitic/hhh/options_files/hhh_v33_SVB.json --gpus 4
date
exit
