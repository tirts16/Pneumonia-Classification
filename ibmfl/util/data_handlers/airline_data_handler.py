"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
from __future__ import print_function
import logging
import numpy as np
import pandas as pd

from ibmfl.util.datasets import load_airline
from sklearn.model_selection import train_test_split
from ibmfl.data.data_handler import DataHandler

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 1234


class AirlineDataHandler(DataHandler):
    """
    Data handler for Airline Delay dataset to train a Multiclass
    Classification Model. TEST_SIZE is set to 0.2, and RANDOM_STATE is set to 42.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config: self.file_name = data_config['txt_file']

    def get_data(self):
        """
        Obtains generated data and splits to test and train sets.

        :param nb_points: Number of data points to be included in each set
        :type nb_points: `int`
        :return: training data
        :rtype: `tuple`
        """
        if self.file_name is None:
            X, y = load_airline()
        else:
            try:
                logger.info('Loaded training data from ' + str(self.file_name))
                data = pd.read_csv(self.file_name, header=None).to_numpy()
                X, y = data[:, :-1], data[:, -1].astype('int')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        return (x_train, y_train), (x_test, y_test)

    def preprocess(self, X, y):
        return X, y