# _*_ coding: utf-8 _*_

"""
Created by Raj Nath Patel on 17/05/18

"""

import logging


class LogManager(object):
    """
    This class contains the logger configurations
    """
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = logging.INFO
    # general logger
    LOG_FILE = "dl4nlp.log"
    logger = logging.getLogger("GeneralLogger")
    logger.setLevel(LOG_LEVEL)
    messaging_logger_file_handler = logging.FileHandler(LOG_FILE)
    messaging_logger_file_handler.setLevel(LOG_LEVEL)
    messaging_logger_file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(messaging_logger_file_handler)