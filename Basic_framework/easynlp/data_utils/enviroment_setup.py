import os
import logging
from logging import handlers
import time
import torch
from pytz import timezone, utc
from datetime import datetime

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, log_filename, level='info', when='D', backCount=3, time_zone="localtime"):
        self.logger = logging.getLogger(log_filename)
        log_format_str = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - '
                                           '%(levelname)s - %(module)s : %(message)s')  # Set format
        console_format_str = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')  # Set format

        if timezone is not "localtime":
            # https://stackoverflow.com/questions/32402502/
            def customTime(*args):
                utc_dt = utc.localize(datetime.utcnow())
                my_tz = timezone(time_zone)
                converted = utc_dt.astimezone(my_tz)
                return converted.timetuple()

            log_format_str.converter = customTime
            console_format_str.converter = customTime

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


def set_logger(args, save_path_name="outputs", log_level="debug", time_zone="Asia/Shanghai"):
    '''
    Set logging file

    :param args: parameters
    :param save_path_name: set the save path
    :param log_level: {debug, info, warning, error, crit}
    :return: {args.timestamp, args.current_save_path, args.log_file, args.logger}
    '''
    if args.if_load_from_checkpoint:
        timestamp = args.checkpoint_name
    else:
        timestamp = str(int(time.time()))
    args.timestamp = timestamp
    args.current_save_path = '{}/{}/'.format(save_path_name, args.timestamp)

    if not os.path.exists(args.current_save_path):
        # print("{} is not exists and is now created".format(args.current_save_path))
        os.makedirs(args.current_save_path)  # Create the output path
    args.log_file = args.current_save_path + time.strftime("log_%Y_%m_%d_%H_%M_%S.txt", time.localtime())
    args.log = Logger(args.log_file, level=log_level, time_zone=time_zone)
    args.logger = args.log.logger


def set_gpu(args):
    if torch.cuda.is_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        args.logger.info("You are now using GPU {}".format(args.gpus))
    else:
        args.logger.warning("CUDA is not avaliable, so now in CPU mode!")






