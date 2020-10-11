"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2020 All Rights Reserved.
"""
import logging
from sklearn.linear_model import SGDClassifier, SGDRegressor
import pickle
import numpy as np

from ibmfl.util import config
from ibmfl.model.sklearn_fl_model import SklearnFLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import LocalTrainingException, \
    ModelInitializationException, ModelException

logger = logging.getLogger(__name__)


class SklearnSGDFLModel(SklearnFLModel):
    """
    Wrapper class for sklearn.linear_model.SGDClassifier and
    sklearn.linear_model.SGDRegressor.
    """

    def __init__(self, model_name, model_spec, sklearn_model=None):
        """
        Create a `SklearnSGDFLModel` instance from a
        sklearn.linear_model.SGDClassifier or a
        sklearn.linear_model.SGDRegressor model.
        If sklearn_model is provided, it will use it; otherwise it will take
        the model_spec to create the model.

        :param model_name: A name specifying the type of model, e.g., \
        linear_SVM
        :type model_name: `str`
        :param model_spec: A dictionary contains model specification
        :type model_spec: `dict`
        :param sklearn_model: Compiled sklearn model
        :type sklearn_model: `sklearn.linear_model`
        """
        super().__init__(model_name, model_spec, sklearn_model)

        if sklearn_model:
            if not issubclass(type(sklearn_model), (SGDClassifier,
                                                    SGDRegressor)):
                raise ValueError('Compiled sklearn model needs to be provided'
                                 '(sklearn.linear_model). '
                                 'Type provided' + str(type(sklearn_model)))

            self.model = sklearn_model

    def fit_model(self, train_data, fit_params=None):
        """
        Fits current model with provided training data.

        :param train_data: Training data a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param fit_params: (Optional) Dictionary with hyperparameters that \
        will be used to call sklearn.linear_model fit function. \
        Provided hyperparameter should only contains parameters that \
        match sklearn expected values, e.g., `learning_rate`, which provides \
        the learning rate schedule. \
        If no `learning_rate` or `max_iter` is provided, a default value will \
        be used ( `optimal` and `1`, respectively).
        :return: None
        """
        # Default values
        max_iter = 1
        warm_start = True

        # Extract x_train and y_train, by default,
        # label is stored in the last column
        x = train_data[0]
        y = train_data[1]
        sample_weight = None

        if 'hyperparams' in fit_params:
            hyperparams = fit_params['hyperparams']
        else:
            hyperparams = None

        if 'sample_weight' in fit_params:
            sample_weight = fit_params['sample_weight']
            if sample_weight.shape[0] != x.shape[0]:
                raise ModelInitializationException(
                    'Number of weights does not match number of samples '
                    'in training set.')

        try:
            training_hp = hyperparams['local']['training']
            if 'max_iter' not in training_hp:
                training_hp['max_iter'] = max_iter
                logger.info('Using default max_iter: ' + str(max_iter))

            # set warm_start to True
            training_hp['warm_start'] = warm_start
            logger.info('Set warm_start as ' + str(warm_start))

            for key, val in training_hp.items():
                self.model.set_params(**{key: val})
        except Exception as e:
            logger.exception(str(e))
            raise LocalTrainingException(
                'Error occurred while setting up model parameters')

        try:
            self.model.fit(x, y, sample_weight=sample_weight)
        except Exception as e:
            logger.info(str(e))
            raise LocalTrainingException(
                'Error occurred while performing model.fit'
            )

    def update_model(self, model_update):
        """
        Update sklearn model with provided model_update, where model_update
        should contains `coef_` and `intercept_` having the same dimension
        as expected by the sklearn.linear_model.
        `coef_` : np.ndarray, shape (1, n_features) if n_classes == 2
        else (n_classes, n_features)
        `intercept_` : np.ndarray, shape (1,) if n_classes == 2 else (n_classes,)

        :param model_update: `ModelUpdate` object that contains the coef_ and \
        the intercept vectors that will be used to update the model.
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            weights = model_update.get('weights')

            if isinstance(self.model, SGDClassifier):
                coef = np.array(weights)[:, :-1]
                intercept = np.array(weights)[:, -1]
            elif isinstance(self.model, SGDRegressor):
                coef = np.array(weights)[:-1]
                intercept = np.array(weights)[-1].reshape(1,)
            else:
                raise LocalTrainingException(
                    "Expecting scitkit-learn model of "
                    "type either "
                    "sklearn.linear_model.SGDClassifier "
                    "or sklearn.linear_model.SGDRegressor."
                    "Instead provided model is of type "
                    + str(type(self.model)))
            try:
                self.model.coef_ = coef
                self.model.intercept_ = intercept
            except Exception as e:
                raise LocalTrainingException('Error occurred during '
                                             'updating the model weights.' +
                                             str(e))
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are: ' +
                                         str(type(model_update)))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """

        coef = self.model.coef_
        intercept = self.model.intercept_
        if isinstance(self.model, SGDClassifier):
            n_classes = np.shape(coef)[0]
            intercept = np.reshape(intercept, [n_classes, 1])
            w = np.append(coef, intercept, axis=1)
        elif isinstance(self.model, SGDRegressor):
            w = np.append(coef, intercept)
        else:
            raise LocalTrainingException("Expecting scitkit-learn model of "
                                         "type either "
                                         "sklearn.linear_model.SGDClassifier "
                                         "or sklearn.linear_model.SGDRegressor."
                                         "Instead provided model is of type "
                                         + str(type(self.model)))

        return ModelUpdate(weights=w.tolist(),
                           coef=self.model.coef_,
                           intercept=self.model.intercept_)

    def evaluate(self, test_dataset, **kwargs):
        """
        Evaluates the model given testing data.
        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, test) or a datagenerator of of type `keras.utils.Sequence`, 
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray`

        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """

        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            return self.evaluate_model(x_test, y_test)

        else:
            raise ModelException("Invalid test dataset!")

    def evaluate_model(self, x, y, **kwargs):
        """
        Evaluates the model given test data x and the corresponding labels y.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Corresponding true labels to x
        :type y: `np.ndarray`
        :param kwargs: Optional sample weights accepted by model.score
        :return: score, mean accuracy on the given test data and labels
        :rtype: `dict`
        """
        acc = {}
        acc['score'] = self.model.score(x, y, **kwargs)
        return acc

    def predict_proba(self, x):
        """
        Perform prediction for the given input.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict_proba(x)

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains the following items: model_spec['model_definition']
        contains the model definition as
        type sklearn.linear_model.SGDClassifier
        or sklearn.linear_model.SGDRegressor.

        :param model_spec: Model specification contains \
        a compiled sklearn model.
        :param model_spec: `dict`
        :return: model
        :rtype: `sklearn.cluster`
        """
        model = None
        try:
            if 'model_definition' in model_spec:
                model_file = model_spec['model_definition']
                model_absolute_path = config.get_absolute_path(model_file)

                with open(model_absolute_path, 'rb') as f:
                    model = pickle.load(f)

                if not issubclass(type(model), (SGDClassifier, SGDRegressor)):
                    raise ValueError('Provided compiled model in model_spec '
                                     'should be of type sklearn.linear_model.'
                                     'Instead they are:' + str(type(model)))
        except Exception as ex:
            raise ModelInitializationException('Model specification was '
                                               'badly form', str(ex))
        return model
