import os
import logging
from logging import handlers
import time
import torch
from torch.utils.data import DataLoader


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, log_filename, level='info', when='D', backCount=3):
        self.logger = logging.getLogger(log_filename)
        log_format_str = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - '
                                           '%(levelname)s - %(module)s : %(message)s')  # Set format
        console_format_str = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')  # Set format
        self.logger.setLevel(self.level_relations.get(level))  # Set log level
        sh = logging.StreamHandler()  # Print to console
        sh.setFormatter(console_format_str)
        # Write to file
        th = handlers.TimedRotatingFileHandler(filename=log_filename, when=when, backupCount=backCount, encoding='utf-8')
        # Create a processor that automatically generates files at specified intervals
        # 'backupCount' is the number of backup files. If it exceeds this number, it will be deleted automatically.
        # 'when' is the time unit of the interval
        th.setFormatter(log_format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)



