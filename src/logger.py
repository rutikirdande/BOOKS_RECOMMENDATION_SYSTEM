import logging

def setup_logger(log_file):
    """
    Set up a logger with a specified log file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
