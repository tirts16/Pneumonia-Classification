"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
import abc
import os
from pathlib import Path
import ibmfl.envs as fl_envs


class FLModel(abc.ABC):
    """
    Base class for all FLModels. This includes supervised and unsupervised ones.
    """

    def __init__(self, model_type, model_spec, **kwargs):
        """
        Initializes an `FLModel` object

        :param model_type: String describing the model e.g., keras_cnn
        :type model_type: `str`
        :param model_spec: Specification of the the model
        :type model_spec: `dict`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        """
        self.model_name = model_type
        self.model_spec = model_spec

    @abc.abstractmethod
    def fit_model(self, train_data, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data.
        :type train_data: Data structure containing training data, \
        varied based on model types.
        :param kwargs: Dictionary of model-specific arguments for fitting, \
        e.g., hyperparameters for local training, information provided \
        by aggregator, etc.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update_model(self, model_update, **kwargs):
        """
        Updates model using provided `model_update`. Additional arguments
        specific to the model can be added through `**kwargs`

        :param model_update: Model with update. This is specific to each model \
        type e.g., `ModelUpdateSGD`. The specific type should be checked by \
        the corresponding FLModel class.
        :type model_update: `ModelUpdate`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_update(self):
        """
        Generate a `ModelUpdate` object specific to the FLModel being trained.
        This object will be shared with other parties.

        :return: Model Update. Object specific to model being trained.
        :rtype: `ModelUpdate`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`

        :return: Predictions
        :rtype: Data structure the same as the type defines labels \
        in testing data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_model(self, x, y, batch_size=128, **kwargs):
        """
        Evaluates model given the samples x and true labels y.
        Multiple evaluation metrics are returned in a dictionary

        :param x: Samples with shape as expected by the model.
        :type x: Data structure as expected by the model
        :param y: Corresponding labels to x
        :type y: Data structure the same as the type defines labels \
        in testing data.
        :param batch_size: batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of model-specific arguments.
        :type kwargs: `dict`
        :return: Dictionary with all evaluation metrics provided by specific \
        implementation.
        :rtype: `dict`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in \
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, filename):
        """
        Load model from provided filename

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, \
        the model will be stored in the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: model
        """
        raise NotImplementedError

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.

        :return: res
        :rtype: `bool`
        """
        raise NotImplementedError

    @staticmethod
    def get_model_absolute_path(filename):
        """Construct absolute path using MODEL_DIR env variable

        :param filename: Name of the file
        :type filename: `str`
        :return: absolute_path; constructed absolute path using model_dir
        :rtype: `str``

        """
        if "MODEL_DIR" in os.environ:
            model_path = Path(os.environ["MODEL_DIR"])
        else:
            model_path = Path(fl_envs.model_directory)

        model_path.mkdir(parents=True, exist_ok=True)
        absolute_path = model_path.joinpath(filename)
        return str(absolute_path)
