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
import abc

from ibmfl.exceptions import FLException, GlobalTrainingException, \
    WarmStartException
from ibmfl.aggregator.metric_service import FLMetricsManager

logger = logging.getLogger(__name__)


class FusionHandler(abc.ABC):

    """
    Base class for Fusion
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_model=None,
                 **kwargs):
        """
        Initializes an `FusionHandler` object with provided fl_model,
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
        :type kwargs: `dict`
        """
        self.name = "Fusion-Algorithm-Name"
        self.ph = protocol_handler
        self.hyperparams = hyperparams
        self.data_handler = data_handler
        self.fl_model = fl_model
        self.metrics_manager = FLMetricsManager()
        self.metrics_party = {}
        self.perc_quorum = 1.
        if hyperparams and hyperparams.get('global') is not None:
            if 'perc_quorum' in hyperparams.get('global'):
                self.perc_quorum = hyperparams.get('global').get('perc_quorum')

        self.warm_start = False

        # load warm start flag if any
        if 'info' in kwargs and kwargs['info'] is not None:
            self.warm_start = kwargs['info'].get('warm_start')
            if not isinstance(self.warm_start, bool):
                logger.info('Warm start flag set to False.')
                self.warm_start = False

    @abc.abstractmethod
    def start_global_training(self):
        """
        Starts global federated learning training process.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_global_model(self):
        """
        Returns the current global model at the aggregator side or
        model parameters that allow parties to reconstruct the global model.
        """
        raise NotImplementedError

    def initialization(self):
        """
        Perform initialization of the global training,
        e.g., warm-start setup etc.

        :return: None
        """
        if self.warm_start and self.fl_model:
            if self.fl_model.is_fitted():
                logger.info('Warm start enabled, '
                            'starting syncing the provided model...')
                try:
                    self.send_global_model()
                except Exception as ex:
                    logger.exception(ex)
                    raise WarmStartException(
                        'Error occurred during syncing the provided model.')
            else:
                raise WarmStartException(
                    'Provided model for warm start is not fitted.')
        elif self.warm_start and not self.fl_model:
            raise WarmStartException(
                'No initial model is provided for warm start process.')
        else:
            logger.info('Warm start disabled.')

    def get_registered_parties(self):
        """
        Returns a list of parties that registered for
        the current federated learning task.

        :return: lst_parties
        :rtype: `list`
        """
        return self.ph.get_available_parties()

    def query(self, function, payload, lst_parties=None, uniform_payload=True):
        """
        Generic query wrapper function to call arbitrary function defined within
        the local training handler of the party. Returns a list of the return
        values from each of the function, irrespective of whether they provide a
        return value or not.

        :param function: Name of function call that is defined within the local \
        training handler
        :type function: `str`
        :param payload: A dictionary which corresponds to the mapping of the \
        necessary contents to pass into the argument provided in the function \
        header. If `uniform_payload` is True, then distributes the same \
        payload across all parties. If not, then each payload is distributed to \
        each worker as defined by order present in the list of dictionaries.
        :type payload: `dict` or `list` (of type `dict`)
        :param lst_parties: List of parties to receive the query. \
        Each entry of the list should be of type `PartyConnection`, and \
        the length of the `lst_parties` should match the length of `payload`. \
        If `lst_parties` is None, by default it will send queries to all \
        parties as defined by `get_registered_parties`.
        :type lst_parties: `list`
        :param uniform_payload: A boolean indicator to determine whether the \
        provided payload is the same across all parties. The default behavior is \
        defined as distributing the same parameter across all parties.
        :type uniform_payload: `boolean`
        :return: response
        :rtype: `list`
        """
        response = []

        try:
            # Check for Parties
            if lst_parties is None:
                lst_parties = self.get_registered_parties()

            # Validate and Construct Deployment Payload
            if uniform_payload:
                if not isinstance(payload, dict):
                    raise FLException('Message content is not in the correct '
                                      'format. Message content should be in the '
                                      'type of dictionary. '
                                      'Instead it is ' + str(type(payload)))

                lst_payload = [{'func': function, 'args': payload}
                               for i in range(len(lst_parties))]
            else:
                if not all(isinstance(x, dict) for x in payload):
                    raise FLException('One or more of the message content is not '
                                      'in the correct format. Message content '
                                      'should be in the type of list of dict.')

                if len(payload) != len(lst_parties):
                    raise FLException('The number of parties does not match '
                                      'lst_parties.')

                lst_payload = [{'func': function, 'args': p} for p in payload]

            response = self.query_parties(lst_payload, lst_parties)

        except Exception as ex:
            logger.exception(str(ex))
            logger.info('Error occurred when sending queries to parties.')

        return response

    def query_all_parties(self, payload):
        """
        Sending queries to all registered parties.
        The query content is provided in `payload`.

        :param payload: Content of a query.
        :type payload: `dict`
        :return: lst_model_updates: a list of replies gathered from \
        the queried parties, each entry of the list should be \
        of type `ModelUpdate`.
        :rtype: `list`
        """
        lst_parties = self.get_registered_parties()
        lst_model_updates = self.query_parties(payload, lst_parties)
        return lst_model_updates

    def query_parties(self, payload, lst_parties):
        """
        Sending queries to the corresponding list of parties.
        The query contents is provided in `payload`.
        The corresponding recipients are provided in `lst_parties`.

        :param payload: Content of a query or contents of multiple queries
        :type payload: `dict` if a single query content will be sent \
        to `lst_parties` or `list` if multiple queries will be sent to \
        the corresponding parties specifying by `lst_parties`.
        :param lst_parties: List of parties to receive the query. \
        Each entry of the list should be of type `PartyConnection`, and \
        the length of the `lst_parties` should match the length of `payload` \
        if multiple queries will be sent.
        :type lst_parties: `list`
        :return: lst_model_updates: a list of replies gathered from \
        the queried parties, each entry of the list should be \
        of type `ModelUpdate`.
        :rtype: `list`
        """
        if lst_parties is None:
            raise FLException('No recipient is provided for the query.')

        lst_model_updates = []
        self.metrics_party = {}
        if hasattr(self, "perc_quorum"):
            perc_q = self.perc_quorum
        else:
            perc_q = 1.
        try:
            if isinstance(payload, dict):

                # send one payload to a list of parties
                id_request = self.ph.query_parties(lst_parties, payload)
                self.ph.periodically_verify_quorum(lst_parties,
                                                   id_request=id_request,
                                                   perc_quorum=perc_q)
                for p in lst_parties:
                    party = self.ph.get_party_by_id(p)
                    if party.get_party_response(id_request):
                        lst_model_updates.append(
                            party.get_party_response(id_request))
                    if party.get_party_metrics(id_request):
                        self.metrics_party[str(p)] = party.get_party_metrics(
                            id_request)
            elif isinstance(payload, list):

                # send multiply payloads to the corresponding lst of parties
                lst_id_request = self.ph.query_parties_data(lst_parties,
                                                            payload)
                self.ph.periodically_verify_quorum(
                    lst_parties,
                    id_request_list=lst_id_request,
                    perc_quorum=perc_q)
                for p in lst_parties:
                    party = self.ph.get_party_by_id(p)
                    if party.get_party_response(lst_id_request[lst_parties.index(p)]):
                        lst_model_updates.append(party.get_party_response(
                            lst_id_request[lst_parties.index(p)]))
                    if party.get_party_metrics(lst_id_request[lst_parties.index(p)]):
                        self.metrics_party[str(p)] = party.get_party_metrics(
                            lst_id_request[lst_parties.index(p)])
            else:
                raise FLException('Message content is not in the correct '
                                  'format. Message content should be in the '
                                  'type of dictionary or '
                                  'list of dictionaries. '
                                  'Instead it is ' + str(type(payload)))
        except Exception as ex:
            logger.exception(str(ex))
            logger.info('Error occurred when sending queries to parties.')
            raise GlobalTrainingException(
                'Error occurred while sending requests to parties')
        return lst_model_updates

    def save_parties_models(self):
        """
        Requests all parties to save local models.
        """
        lst_parties = self.ph.get_available_parties()

        data = {}
        id_request = self.ph.save_model_parties(lst_parties, data)
        logger.info('Finished saving the models.')

    def save_local_model(self, filename=None):
        """Save aggregated model locally
        """
        saved_file = None
        if self.fl_model:
            saved_file = self.fl_model.save_model(filename=filename)

        return saved_file

    def evaluate_model(self):
        """
        Requests all parties to send model evaluations.
        """
        lst_parties = self.ph.get_available_parties()
        # TODO: parties should return evaluation results to aggregator
        data = {}
        id_request = self.ph.eval_model_parties(lst_parties, data)
        logger.info('Finished evaluate model requests.')

    def send_global_model(self):
        """
        Send global model to all the parties
        """
        # Select data parties
        lst_parties = self.ph.get_available_parties()

        model_update = self.get_global_model()
        payload = {'model_update': model_update
                   }

        logger.info('Sync Global Model' + str(model_update))
        self.ph.sync_model_parties(lst_parties, payload)

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler
        Includes all the the variables required to bring back fusion handler
        to the current state.
        """
        raise NotImplementedError

    def save_current_state(self):
        """Save current fusion handler state using metrics manager. Save current model,
        collect metrics and use metricsmanager to save them.
        """
        metrics = {}
        fusion_metrics = self.get_current_metrics()
        metrics['fusion'] = fusion_metrics
        metrics['party'] = self.metrics_party
        #model_file = self.save_local_model()
        #metrics['model_file'] = model_file

        self.metrics_manager.save_metrics(metrics)
