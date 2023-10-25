# logger.py
import logging
import os


def setup_logger():
    # Create a custom format for your logs
    log_format = "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %(levelname)s: %(message)s"

    # Create a log handler for file output
    path = os.path.join(os.path.dirname(
        __file__), '../../../logger/debug.log')
    file_handler = logging.FileHandler(filename=path, mode='w')
    # file_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Apply the custom format to the handler
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    # Create a logger and add the handler
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
