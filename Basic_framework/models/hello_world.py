# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class HELLO(nn.Module):
    def __init__(self):
        super(HELLO, self).__init__()
        self.init_info = "Hello World!"
        print(self.init_info)

    def forward(self):
        pass
        return

