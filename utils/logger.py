import logging
import logging.handlers
import os
from datetime import datetime

LOG_PATH = "./logs/"
LOG_EXT = ".log"
FILE_HANDLER_FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s"


def get_current_time():
    today = datetime.now()
    date_time = today.strftime("%Y%m%d_%H%M%S")
    return date_time


def remove_handler(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def set_logger(log_file_name):
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)

    curr_time = get_current_time()
    log_file = log_file_name + "_" + curr_time + LOG_EXT
    log_file_path = os.path.join(LOG_PATH, log_file)

    remove_handler(logger)
    file_handler = logging.FileHandler(
        log_file_path, mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))
    logger.addHandler(file_handler)

    return logger


def get_logger_path(logger):
    log_file_path = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path = handler.baseFilename
            break
    else:
        log_file_path = "Invalid Path"

    return log_file_path
