import os
from datetime import datetime
import logging
import logging.handlers


LOG_PATH = './logs/'
LOG_EXT = 'log'
FILE_HANDLER_FORMAT = '[%(asctime)s][%(levelname)s]: %(message)s'


def get_current_time():
    today = datetime.now()
    date_time = today.strftime("%Y%m%d_%H%M%S")
    return date_time


def set_logger(name):
    logging.basicConfig(level='NOTSET')
    logger = logging.getLogger(name)

    curr_time = get_current_time()
    log_file_path = os.path.join(LOG_PATH, name, curr_time, LOG_EXT)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)

    return logger
