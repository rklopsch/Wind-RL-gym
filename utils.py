# utils.py
import os

def is_verbose():
    return os.getenv('WINDRL_VERBOSE', 'false').lower() in ['true', '1', 'yes']

