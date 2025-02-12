"""
Dummy helper functions.
"""

import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DummyLogger")
    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger is set up and working!")
