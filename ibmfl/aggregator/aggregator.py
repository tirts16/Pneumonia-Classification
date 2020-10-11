"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
#!/usr/bin/env python3

"""
Aggregator is an application which will allow users
to execute controlled Federated Learning tasks
"""
import re
import os
import sys
import logging

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.aggregator.states import States
from ibmfl.util.config import configure_logging_from_file, \
    get_aggregator_config
from ibmfl.connection.route_declarations import get_aggregator_router
from ibmfl.connection.router_handler import Router

logger = logging.getLogger(__name__)


class Aggregator(object):
    """
    Aggregator class to create an aggregator application
    """

    def __init__(self, **kwargs):
        """
        Initializes an `Aggregator` object

        :param config_file: path to yaml file containing configuration
        :type config_file: `str`
        """
        configure_logging_from_file()

        cls_config = get_aggregator_config(**kwargs)

        self.data_handler = None
        self.fl_model = None

        data_config = cls_config.get('data')
        model_config = cls_config.get('model')
        connection_config = cls_config.get('connection')
        ph_config = cls_config.get('protocol_handler')
        fusion_config = cls_config.get('fusion')
        max_timeout = None

        try:
            # Load data (optional field)
            # - in some cases the aggregator doesn't have data for testing purposes
            if data_config:
                data_cls_ref = data_config.get('cls_ref')
                data_info = data_config.get('info')
                self.data_handler = data_cls_ref(data_config=data_info)

            # Read and create model (optional field)
            # In some cases aggregator doesn't need to load the model:
            if model_config:
                model_cls_ref = model_config.get('cls_ref')
                spec = model_config.get('spec')
                self.fl_model = model_cls_ref('', spec)

            # Load hyperparams
            self.hyperparams = cls_config.get('hyperparams')

            connection_cls_ref = connection_config.get('cls_ref')
            connection_info = connection_config.get('info')
            connection_synch = connection_config.get('sync')
            if 'max_timeout' in cls_config.get('hyperparams').get('global'):
                max_timeout = cls_config.get('hyperparams').get(
                    'global').get('max_timeout')

            self.connection = connection_cls_ref(connection_info)
            self.connection.initialize_sender()

            ph_cls_ref = ph_config.get('cls_ref')
            self.proto_handler = ph_cls_ref(self.connection.sender,
                                            connection_synch,
                                            max_timeout)

            self.router = Router()
            get_aggregator_router(self.router, self.proto_handler)

            fusion_cls_ref = fusion_config.get('cls_ref')
            fusion_info = fusion_config.get('info')
            self.fusion = fusion_cls_ref(self.hyperparams,
                                         self.proto_handler,
                                         data_handler=self.data_handler,
                                         fl_model=self.fl_model,
                                         info=fusion_info)

            self.connection.initialize_receiver(router=self.router)

        except Exception as ex:
            logger.info(
                'Error occurred while loading aggregator configuration')
            logger.exception(ex)

        else:
            logger.info("Aggregator initialization successful")

    def start(self):
        """
        Start a server for the aggregator in a new thread
        Parties can connect to register

        """
        try:
            self.connection.start()
        except Exception as ex:
            logger.error("Error occurred during start")
            logger.error(ex)
        else:
            logger.info("Aggregator start successful")

    def stop(self):
        """
        Stop the aggregator server

        :param: None
        :return: None
        """
        try:
            self.proto_handler.stop_parties()
            self.connection.stop()
        except Exception as ex:
            logger.error("Error occurred during stop")
            logger.error(ex)
        else:
            logger.info("Aggregator stop successful")

    def start_training(self):
        """
        Start federated learning training. Request all the registered
        parties to initiate training and send model update

        :param: None
        :return: Boolean
        :rtype: `boolean`
        """
        logger.info('Initiating Global Training.')
        try:
            self.fusion.initialization()
        except Exception as ex:
            logger.exception('Exception occurred during the initialization '
                             'of the global training.')
            logger.exception(ex)
            return False
        try:
            self.fusion.start_global_training()
        except Exception as ex:
            logger.exception('Exception occurred while training.')
            logger.exception(ex)
            return False
        else:
            logger.info('Finished Global Training')
        return True

    def save_model(self):
        """
        Request all parties to save models
        """
        logger.info('Initiating save model request.')
        try:
            self.fusion.save_parties_models()
        except Exception as ex:

            logger.exception(ex)
        else:
            logger.info('Finished save requests')

    def eval_model(self):
        """
        Request all parties to print evaluations
        """
        logger.info('Initiating evaluation requests.')
        try:
            self.fusion.evaluate_model()

        except Exception as ex:
            logger.exception('Exception occurred during party evaluations.')
            logger.exception(ex)
        else:
            logger.info('Finished eval requests')

    def model_synch(self):
        """
        Send global model to the parties
        """
        logger.info('Initiating global model sync requests.')
        try:
            self.fusion.send_global_model()
        except Exception as ex:
            logger.exception('Exception occurred during sync model.')
            logger.exception(ex)
        else:
            logger.info('Finished sync model requests')


if __name__ == '__main__':
    """
    Main function can be used to create an application out
    of our Aggregator class which could be interactive
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        logging.error('Please provide yaml configuration')

    server_process = None
    config_file = sys.argv[1]

    if not os.path.isfile(config_file):
        logging.error("config file '{}' does not exist".format(config_file))

    agg = Aggregator(config_file=config_file)
    # Indefinite loop to accept user commands to execute
    while 1:
        msg = sys.stdin.readline()
        # TODO: move it to Aggregator
        if re.match('START', msg):
            agg.proto_handler.state = States.CLI_WAIT
            logging.info("State: " + str(agg.proto_handler.state))
            # Start server
            agg.start()

        elif re.match('STOP', msg):
            agg.proto_handler.state = States.PROC_STOP
            logging.info("State: " + str(agg.proto_handler.state))
            agg.stop()
            break

        elif re.match('TRAIN', msg):
            agg.proto_handler.state = States.PROC_TRAIN
            logging.info("State: " + str(agg.proto_handler.state))
            success = agg.start_training()
            if not success:
                agg.stop()
                break

        elif re.match('SAVE', msg):
            agg.proto_handler.state = States.PROC_SAVE
            logging.info("State: " + str(agg.proto_handler.state))
            agg.save_model()

        elif re.match('EVAL', msg):
            agg.proto_handler.state = States.PROC_EVAL
            logging.info("State: " + str(agg.proto_handler.state))
            agg.eval_model()

        elif re.match('SYNC', msg):
            agg.proto_handler.state = States.PROC_SYNC
            logging.info("State: " + str(agg.proto_handler.state))
            agg.model_synch()

    exit()
