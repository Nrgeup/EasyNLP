# -*- coding: utf-8 -*-
# Author: Ke Wang
# Contact: wangke17[AT]pku.edu.cn

######################################################################################
#  Import Packages
######################################################################################
# Basic Packages
import os
import time
import argparse
import math
import numpy
import torch
import torch.nn as nn
# import matplotlib
# from matplotlib import pyplot as plt


# Import your custom models
import easynlp.models as Models
import easynlp.data_utils.enviroment_setup as EnviromentSetup


######################################################################################
#  Parsing parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
'''parser.add_argument(
        "--task_name", 
        default=None, 
        type=str/int/float/bool, 
        required=True, 
        choices=[], 
        help="The name of the task to train.")
'''
# -----------------  Environmental parameters  ----------------- #
parser.add_argument('--gpus', type=str, default='0', help='Specifies the GPU to use.')
parser.add_argument('--if_load_from_checkpoint', type=bool, default=False, help='If load from saved checkpoint.')

# -----------------  File parameters  ----------------- #
parser.add_argument('--checkpoint_name', type=str, default="None", help='Saved checkpoint name.')

# -----------------  Hyper parameters  ----------------- #



hparams = parser.parse_args()
######################################################################################
#  End of hyper parameters
######################################################################################


def pre_setup():
    # Enviroment setup
    EnviromentSetup.set_logger(hparams)
    EnviromentSetup.set_gpu(hparams)
    hparams.logger.info("Enviroment setup success!")


def train():
    pass


def test():
    pass


if __name__ == '__main__':
    pre_setup()
