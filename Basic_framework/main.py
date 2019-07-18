# -*- coding: utf-8 -*-
# Author: Ke Wang
# Contact: wangke17[AT]pku.edu.cn

######################################################################################
#  Packages
######################################################################################
# Basic Packages
import os
import logging
import time
import argparse
import math
import numpy
import torch
import torch.nn as nn
import matplotlib
from matplotlib import pyplot as plt

# Import your custom models.
from models.hello_world import HELLO
from data_utils import get_cuda


# Pre-set
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
######################################################################################
#  Hyper-parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
# parser.add_argument("--task_name",
#                     default=None,
#                     type=str/int/float/bool,
#                     required=True,
#                     choices=[],
#                     help="The name of the task to train.")

# Environmental parameters
parser.add_argument('--gpu_id', type=str, default='0', help='Specifies the GPU to use.')
parser.add_argument('--if_load_from_checkpoint', type=bool, default=False, help='If load from saved checkpoint.')

#  File parameters
parser.add_argument('--checkpoint_name', type=str, default="None", help='Saved checkpoint name.')

#  Model parameters

######################################################################################
#  End of hyper parameters
######################################################################################
args = parser.parse_args()
# Set logging file
if args.if_load_from_checkpoint:
    timestamp = args.checkpoint_name
else:
    timestamp = str(int(time.time()))
args.current_save_path = 'outputs/%s/' % timestamp
if not os.path.exists(args.current_save_path):
    os.makedirs(args.current_save_path)   # Create the output path
args.log_file = args.current_save_path + time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", time.localtime())
logging.basicConfig(level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    filename=args.log_file)  # info()/debug()/warning()
args.logger = logging.getLogger(__name__)
# set gpu
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.logger.info("You are now using GPU {}".format(args.gpu_id))
else:
    args.logger.warning("CUDA is not avaliable, so now in CPU mode!")

# Write your main code here.
if __name__ == '__main__':
    HELLO()
