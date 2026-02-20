import logging
import os
import sys


def setup_root_logger():
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        # set level based on environment variable or default to INFO
        if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG":
            root.setLevel(logging.DEBUG)
        else:
            root.setLevel(logging.INFO)
        root.addHandler(handler)


# Configure on import
setup_root_logger()
