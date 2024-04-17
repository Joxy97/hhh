import numpy as np
import torch
import h5py
import onnxruntime
import json
from itertools import permutations

with h5py.File("/eos/user/j/jmitic/hhh/data/svb_testing.h5", 'r+') as file:
    
    print(file["CLASSIFICATIONS"]["EVENT"]["signal"])




