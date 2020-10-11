"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
#!/usr/bin/env python3
import re
import os
import sys
import logging

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.party.status_type import StatusType
from ibmfl.util.config import configure_logging_from_file, \
    get_party_config
from ibmfl.connection.route_declarations import get_party_router
from ibmfl.connection.router_handler import Router
from ibmfl.message.message import Message
from ibmfl.message.message_type import MessageType
from ibmfl.exceptions import FLException

logger = logging.getLogger(__name__)


class Party:
    """
    Application that runs FL at the party side. Given a config file, it
    spins a server, creates FLModel of the type that will be trained,
    creates an instance of the DataHandler to be used.
    """

    def __init__(self, **kwargs):
        """
        Initiates a party based on the config_file provided

        :param config_file: path to yml file with the configuration of \
        the party
        :type config_file: `str`
        :return: None
        """
        configure_logging_from_file()

        cls_config = get_party_config(**kwargs)

        self.data_handler = None
        self.fl_model = None

        data_config = cls_config.get('data')
        model_config = cls_config.get('model')
        connection_config = cls_config.get('connection')
        ph_config = cls_config.get('protocol_handler')
        lt_config = cls_config.get('local_training')
        try:
            # Load data (optional field)
            # - in some cases the aggregator doesn't have data for testing purposes
            data_cls_ref = data_config.get('cls_ref')
            data_info = data_config.get('info')
            self.data_handler = data_cls_ref(data_config=data_info)

            # Read and create model (optional field)
            # In some cases aggregator doesn't need to load the model:
            model_cls_ref = model_config.get('cls_ref')
            spec = model_config.get('spec')
            self.fl_model = model_cls_ref('', spec)

            # Load hyperparams
            self.hyperparams = cls_config.get('hyperparams')
            self.agg_info = cls_config.get('aggregator')

            connection_cls_ref = connection_config.get('cls_ref')
            self.connection_info = connection_config.get('info')
            connection_synch = connection_config.get('sync')
            self.connection = connection_cls_ref(self.connection_info)
            self.connection.initialize_sender()

            lt_cls_ref = lt_config.get('cls_ref')
            lt_info = lt_config.get('info')
            self.local_training_handler = lt_cls_ref(self.fl_model,
                                                     self.data_handler,
                                                     info=lt_info)

            ph_cls_ref = ph_config.get('cls_ref')

            self.proto_handler = ph_cls_ref(self.fl_model,
                                            self.connection.sender,
                                            self.data_handler,
                                            self.local_training_handler,
                                            agg_info=self.agg_info,
                                            synch=connection_synch)

            self.router = Router()
            get_party_router(self.router, self.proto_handler)

            self.connection.initialize_receiver(router=self.router)

        except Exception as ex:
            logger.info('Error occurred '
                        'while loading party configuration')
            logger.exception(ex)
        else:
            logger.info("Party initialization successful")

    def register_party(self):
        """
        Registers party
        """
        logger.info('Registering party...')
        register_message = Message(
            MessageType.REGISTER.value, data=self.connection.get_connection_config())
        try:
            response = self.connection.sender.send_message(
                self.agg_info, register_message)
        except FLException as ex:
            logger.exception("Error occurred while registring party!")
            logger.exception(str(ex))
            return

        if response.get_data()['status'] == 'success':
            logger.info('Registration Successful')
        else:
            logger.error('Registration Failed')

    def start(self):
        """
        Start a server for the party in a new thread
        Parties can connect to the server to register

        :param: None
        :return: New process to launch server
        :rtype: `Process`
        """
        try:
            self.connection.start()
        except Exception as ex:
            logger.error("Error occurred during start")
            logger.error(ex)
        else:
            logger.info("Party start successful")

    def stop(self):
        """
        Stop the party server

        :param: None
        :return: None
        """
        try:
            self.connection.stop()

        except Exception as ex:
            logger.error("Error occurred during stop")
            logger.error(ex)
        else:
            logger.info("Party stop successful")

    def evaluate_model(self):
        """
        Calls function that evaluates current model with local testing data
        and prints the results.

        :return: None
        """
        self.proto_handler.print_evaluate_local_model()


if __name__ == '__main__':
    """
    Main function can be used to create an application out
    of our party class which could be interactive
    """
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        logging.error('Please provide yaml configuration')
    config_file = sys.argv[1]
    p = Party(config_file=config_file)

    # Indefinite loop to accept user commands to execute
    while 1:
        msg = sys.stdin.readline()
        if re.match('START', msg):
            # Start server
            p.start()

        if re.match('STOP', msg):
            p.connection.stop()
            break

        if re.match('REGISTER', msg):
            p.register_party()

        if re.match('EVAL', msg):
            p.evaluate_model()

        if p.proto_handler.status == StatusType.STOPPING:
            p.stop()
            break
