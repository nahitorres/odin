import logging
import os
from datetime import datetime
import csv, sys
from types import FunctionType
import numpy as np
from IPython import get_ipython
from PIL import Image
from io import BytesIO
from base64 import b64encode


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()

    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level,
            handlers=[
                logging.StreamHandler()
            ]
        )

    logger.setLevel(log_level)
    return logger


def read_csv_file(path_to_file):
    try:
        with open(path_to_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                line_count += 1
            csv_file.close()
    except:
        print("Unexpected error opening file:", sys.exc_info()[0])
        raise


def verify_path_or_create_directory(path_to_directory, logger=None):
    '''
    Create all the directory (even intermediate ones if they do not exist)
    :param path_to_directory: whole path
    :param logger: if passed, prints logs into Logger instance
    :return:
    '''
    if not os.path.exists(path_to_directory):
        if logger is not None:
            logger.info('Creating a directory in path {}'.format(path_to_directory))
        else:
            print('Creating a directory in path {}'.format(path_to_directory))
        os.makedirs(path_to_directory, exist_ok=True)
    return


def remove_vowels(word):
    without_vw = word
    vowels = ('a', 'e', 'i', 'o', 'u')
    for x in without_vw.lower():
        if x in vowels:
            without_vw = without_vw.replace(x, "")
    return without_vw

def get_min_val(s, fname, miny):
    if miny is None:
        miny = float("inf")

    if not isinstance(s,np.ndarray) and s == fname:
        return min(miny, s)

    elif isinstance(s,list) or isinstance(s,np.ndarray):
        names = np.arange(len(s))
    else:
        names = list(s.keys())

    for name in names:
        if not (isinstance(s[name], dict) or isinstance(s[name],list) or isinstance(s[name], np.ndarray) or name== fname):
            continue

        if name==fname:
            miny = min(miny, s[name])
        else:
            miny = min(miny, get_min_val(s[name], fname, miny))

    return miny


def get_max_val(s, fname, maxy):
    if maxy is None:
        maxy = float("-inf")

    if not isinstance(s,np.ndarray) and s == fname:
        return max(maxy, s)

    elif isinstance(s,list) or isinstance(s,np.ndarray):
        names = np.arange(len(s))
    else:
        names = list(s.keys())

    for k_name in names:
        if not (isinstance(s[k_name], dict) or isinstance(s[k_name],list) or isinstance(s[k_name], np.ndarray) or k_name== fname):
            continue

        if k_name==fname:
            maxy = max(maxy, s[k_name])
        else:
            maxy = max(maxy, get_max_val(s[k_name], fname, maxy))

    return maxy


def methods_from_class(cls):
    return [x for x, y in cls.__dict__.items() if type(y) == FunctionType]


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    

def get_leaflet_url(image): # Obtain BLOB to be used as url in leaflet classes
    ext = image.format # Retrieve image format
    f = BytesIO()
    image.save(f, ext)
    data = b64encode(f.getvalue())
    data = data.decode('ascii')
    leaflet_url = 'data:image/{};base64,'.format(ext) + data #BLOB url
    return leaflet_url
