import os
import time
import random
import numpy as np
import torch
#from torchsummary import summary
from torchsummary import summary
from MST_sunet import mst
import more_itertools as mit
import torchvision.transforms as T
from torch.utils.data import DataLoader
import PIL
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as F
import math
from scipy.signal import butter, sosfiltfilt, resample, stft, butter, sosfiltfilt, welch
import timm
from utils.utils_sunet_folds import create_datasets
import argparse
import heartpy as hp

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str,required=True)   # dataset name
parser.add_argument('-hm','--howmany', type=str,required=True) # how many folds
parser.add_argument('-ts','--trainstride', type=str,required=True) # what is the stride, should be seqlen if now overlapping
parser.add_argument('-sl','--seqlen', type=str,required=True) # sequence lenght, 576
args = parser.parse_args()

dataset = args.data
howmany = str(args.howmany)
seq_len = int(args.seqlen)
train_stride = int(args.trainstride)

BATCH_SIZE = 8
NUM_WORKERS = 0

train_dataset, valid_dataset = create_datasets(dataset,howmany,train_stride=train_stride,seq_len=seq_len,train_temp_aug=False)
print(len(train_dataset))
print(len(valid_dataset))
