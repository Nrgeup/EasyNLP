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

# My model
import data
import model


######################################################################################
#  Hyper-parameters
######################################################################################
parser = argparse.ArgumentParser()
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

_device = torch.device("cuda" if args.cuda else "cpu")







