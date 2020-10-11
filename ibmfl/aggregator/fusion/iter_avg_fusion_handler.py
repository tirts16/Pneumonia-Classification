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

from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.fusion_handler import FusionHandler

logger = logging.getLogger(__name__)


class IterAvgFusionHandler(FusionHandler):
    """
    Class for iterative averaging based fusion algorithms.
    An iterative fusion algorithm here referred to a fusion algorithm that
    sends out queries at each global round to registered parties for
    information, and use the collected information from parties to update
    the global model.
    The type of queries sent out at each round is the same. For example,
    at each round, the aggregator send out a query to request local model's
    weights after parties local training ends.
    The iterative algorithms can be terminated at any global rounds.

    In this class, the aggregator requests local model's weights from all
    parties at each round, and the averaging aggregation is performed over
    collected model weights. The global model's weights then are updated by
    the mean of all collected local models' weights.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an IterAvgFusionHandler object with provided information,
        such as protocol handler, fl_model, data_handler and hyperparams.

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
        :return: None
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_model,
                         **kwargs)
        self.name = "Iterative-Weight-Average"
        self.params_global = hyperparams.get('global') or {}
        self.params_local = hyperparams.get('local') or None

        self.rounds = self.params_global.get('rounds') or 1
        self.curr_round = 0
        self.global_accuracy = -1
        self.termination_accuracy = self.params_global.get(
            'termination_accuracy')

        if fl_model and fl_model.is_fitted():
            model_update = fl_model.get_model_update()
        else:
            model_update = None

        self.current_model_weights = \
            model_update.get('weights') if model_update else None

    def start_global_training(self):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = 0
        while not self.reach_termination_criteria(self.curr_round):
            # construct ModelUpdate
            if self.current_model_weights:
                model_update = ModelUpdate(weights=self.current_model_weights)
            else:
                model_update = None

            payload = {'hyperparams': {'local': self.params_local},
                       'model_update': model_update
                       }
            logger.info('Model update' + str(model_update))

            # query all available parties
            lst_replies = self.query_all_parties(payload)

            self.update_weights(lst_replies)

            # Update model if we are maintaining one
            if self.fl_model is not None:
                self.fl_model.update_model(
                    ModelUpdate(weights=self.current_model_weights))

            self.curr_round += 1
            self.save_current_state()

    def update_weights(self, lst_model_updates):
        """
        Update the global model's weights with the list of collected
        model_updates from parties.
        In this method, it calls the self.fusion_collected_response to average
        the local model weights collected from parties and update the current
        global model weights by the results from self.fusion_collected_response.

        :param lst_model_updates: list of model updates of type `ModelUpdate` to be averaged.
        :type lst_model_updates: `list`
        :return: None
        """
        self.current_model_weights = self.fusion_collected_responses(
            lst_model_updates)

    def fusion_collected_responses(self, lst_model_updates, key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the values (indicating by the key)
        included in each model_update, it finds the mean.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates:  `list`
        :param key: A key indicating what values the method will aggregate over.
        :type key: `str`
        :return: results after aggregation
        :rtype: `list`
        """
        v = []
        for update in lst_model_updates:
            v.append(np.array(update.get(key)))
        results = np.mean(np.array(v), axis=0)

        return results.tolist()

    def reach_termination_criteria(self, curr_round):
        """
        Returns True when termination criteria has been reached, otherwise
        returns False.
        Termination criteria is reached when the number of rounds run reaches
        the one provided as global rounds hyperparameter.
        If a `DataHandler` has been provided and a targeted accuracy has been
        given in the list of hyperparameters, early termination is verified.

        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :return: boolean
        :rtype: `boolean`
        """

        # Only evaluate early termination when it is possible/required:
        if self.data_handler and self.fl_model and self.termination_accuracy:
            (_, _), test_data = self.data_handler.get_data()
            eval_results = self.fl_model.evaluate(test_data)

            if 'acc' in eval_results:
                logger.info('Checking if expected accuracy is reached')
                self.global_accuracy = eval_results['acc']
                if eval_results['acc'] > self.termination_accuracy:
                    logger.info('Early termination reached after running '
                                + str(curr_round) + 'rounds')
                    return True
            else:
                logger.warning('Early termination could not be evaluated '
                               'because fl_model did not return accuracy '
                               'metric')
        if curr_round >= self.rounds:
            logger.info('Reached maximum global rounds. Finish training :) ')
            return True

        return False

    def get_global_model(self):
        """
        Returns last model_update

        :return: model_update
        :rtype: `ModelUpdate`
        """
        return ModelUpdate(weights=self.current_model_weights)

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler

        :return: metrics
        :rtype: `dict`
        """
        fh_metrics = {}
        fh_metrics['rounds'] = self.rounds
        fh_metrics['curr_round'] = self.curr_round
        fh_metrics['acc'] = self.global_accuracy
        #fh_metrics['model_update'] = self.model_update
        return fh_metrics
