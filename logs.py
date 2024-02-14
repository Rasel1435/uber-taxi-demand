import logging

def configure_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%d-%b(%m)-%Y %I:%M:%S'
    )

    return logging.getLogger(__name__)
