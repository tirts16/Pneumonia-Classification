"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
"""
Module providing utility functions for loading datasets for use in FL
"""
import os
import sys
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests


def save_file(path, url):
    """
    Saves a file from URL to path

    :param path: the path to save the file
    :type path; `str`
    :param url: the link to download from
    :type url: `str`
    """
    with requests.get(url, stream=True, verify=False) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def load_xray(normalize=True, download_dir=""):
    """
    Download MNIST training data from source used in `keras.datasets.load_mnist`
    :param normalize: whether or not to normalize data
    :type normalize: bool
    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: 2 tuples containing training and testing data respectively
    :rtype (`np.ndarray`, `np.ndarray`), (`np.ndarray`, `np.ndarray`)
    """
    local_file = os.path.join(download_dir, "xray.npz")
    if not os.path.isfile(local_file):
        print("xray dataset not found")

    with np.load(local_file, allow_pickle=True) as xray:
        x_train, y_train = xray['x_train'], xray['y_train']
        x_test, y_test = xray['x_test'], xray['y_test']
        if normalize:
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')

            x_train /= 255
            x_test /= 255

    return (x_train, y_train), (x_test, y_test)

