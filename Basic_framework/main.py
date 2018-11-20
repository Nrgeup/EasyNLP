# coding: utf-8
# requirements: pytorch: 0.04
# Author: Ke Wang
# Contact: wangke17[AT]pku.edu.cn
import time
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from torch import optim
import numpy
import matplotlib
from matplotlib import pyplot as plt


# Import your model files.
import data
import model


######################################################################################
#  Hyper-parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
# Add your arguments here.
# Example:
# parser.add_argument('--data', type=str, default='../../Data/wikitext-2',
#                     help='location of the data corpus')


parser.add_argument('--device', type=str, default='cpu', help='')

args = parser.parse_args()


# set gpu
if torch.cuda.is_available():
    args.device = "cuda"
    print("Info: You are now using GPU mode:", args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
else:
    print("Warning: You do not have a CUDA device, so you now running with CPU!")


# Write your main code here.









