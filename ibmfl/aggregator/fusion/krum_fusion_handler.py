"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
"""
Module to where fusion algorithms are implemented.
"""
import logging
import numpy as np

from ibmfl.aggregator.fusion.iter_avg_fusion_handler import \
    IterAvgFusionHandler
from ibmfl.exceptions import GlobalTrainingException

logger = logging.getLogger(__name__)


class KrumFusionHandler(IterAvgFusionHandler):
    """
    Class for Krum Fusion.

    Implements the Krum algorithm presented
    here: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an KrumAvgFusionHandler object with provided fl_model,
        data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        """

        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.byzantine_threshold = hyperparams['global']['byzantine_threshold']
        self.name = "Krum"
        self._eps = 1e-6

    def fusion_collected_responses(self, lst_model_updates, key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the weights included in each model_update, it
        finds the best model for the next round.

        :param lst_model_updates: List of model updates of type `ModelUpdate`
        :type lst_model_updates:  `lst`
        :param key: The key we wish to access from the model update
        :return: Result after fusion
        :rtype: `list`
        """
        weights = None

        num_updates = len(lst_model_updates)
        if num_updates <= 2 * self.byzantine_threshold + 2:
            logging.error('KRUM Fusion: Byzantine resilience assumes n > 2*f + 2. '
                          'Please pick the parameters appropriately\n'
                          'n: {}\n'
                          'f: {}\n'.format(len(lst_model_updates), self.byzantine_threshold))
            raise GlobalTrainingException
        else:
            distance = self.get_distance(lst_model_updates, key)
            # score is computed using n-f-2 closest vectors
            th = num_updates - self.byzantine_threshold - 2
            scores = self.get_scores(distance, th)
            selected_idx = np.argmin(scores)

            weights = lst_model_updates[selected_idx].get(key)
            return weights

    def get_distance(self, lst_model_updates, key):
        """
        Generates a matrix of distances between each of the model updates
        to all of the other model updates 

        :param lst_model_updates: List of model updates participating in fusion round
        :type lst_model_updates: `list`
        :param key: Key to pull from model update (default to 'weights')
        :return: distance
        :rtype: `np.array`
        """
        num_updates = len(lst_model_updates)
        distance = np.zeros((num_updates, num_updates), dtype=float)

        lst_model_updates_flattened = []
        for update in lst_model_updates:
            lst_model_updates_flattened.append(
                self.flatten_model_update(update.get(key)))

        for i in range(num_updates):
            curr_vector = lst_model_updates_flattened[i]
            for j in range(num_updates):
                if j is not i:
                    distance[i, j] = np.square(np.linalg.norm(
                        curr_vector - lst_model_updates_flattened[j]))
                    # Default is L-2 norm
        return distance

    @staticmethod
    def flatten_model_update(lst_layerwise_wts):
        """
        Generates a flattened np array for all of the layerwise weights of an update

        :param lst_layerwise_wts: List of layer weights
        :type lst_layerwise_wts: `list`
        :return: `np.array`
        """
        wt_vector = []
        for w in lst_layerwise_wts:
            t = w.flatten()
            wt_vector = np.concatenate([wt_vector, t])
        return wt_vector

    @staticmethod
    def get_scores(distance, th):
        """
        Sorts the distances in an ordered list and returns the list for use to 
        the fusion_collected_responses function

        :param distance: List of distance vector
        :type distance: `list`
        :param th: Threshold
        :return: list of summation of distances
        :rtype: `list`
        """
        distance.sort(axis=1)
        # the +1 is added to account for the zero entry (distance from itself)
        return np.sum(distance[:, 0:th+1], axis=1)
