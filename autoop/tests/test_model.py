import unittest

import numpy as np

from autoop.core.ml.model import MultipleLinearRegression, get_model
from autoop.core.ml.model.classification import (
    WrapKNearestNeighbors,
    WrapNaiveBayes,
    WrapRandomForest,
)
from autoop.core.ml.model.model import Model, ParametersDict
from autoop.core.ml.model.regression import WrapElasticNet, WrapLogisticRegression


class ConcreteModel(Model):
    def fit(self, data) -> None:
        pass

    def predict(self, data) -> None:
        pass


class TestModel(unittest.TestCase):
    def test_parameters_dict(self):
        param_dict = ParametersDict({})
        param_dict.update({"a": 1, "b": 2})
        param_dict.update({"a": 0.1, "b": 0.3})

        self.assertEqual(
            ParametersDict({"a": 0.1, "b": 0.3})._get_keys_as_list(),
            param_dict._get_keys_as_list()
        )

    def test_set_update_wrongly(self):
        param_dict = ParametersDict({"a": 1, "b": 2})
        with self.assertRaises(AttributeError) as cm:
            param_dict.update({"WRONG KEY": 42})

        exception = cm.exception
        self.assertIn(
            member=("ParametersDict.update expects another ParametersDict "
                    "with the same keys"),
            container=str(exception),
        )

    def test_to_and_from_artifact(self):
        model = ConcreteModel(
            type="numerical",
            hyper_parameters={"learning rate": 0.005, "batch size": 10},
            parameters={"slope": 0.5, "intercept": 2}
        )

        artifact = model.to_artifact(name="test artifact")
        recovered_model = ConcreteModel.from_artifact(artifact)

        self.assertEqual(model.type, recovered_model.type)
        self.assertEqual(model.parameters, recovered_model.parameters)
        self.assertEqual(model.hyper_parameters, recovered_model.hyper_parameters)

        # This test tested:
        # - initializer
        # - getters

    def test_set_hyper_parameters(self):
        model = ConcreteModel(
            type="numerical",
            hyper_parameters={"learning rate": 0.005, "batch size": 10},
            parameters={"slope": 0.5, "intercept": 2}
        )
        model.hyper_parameters = {
            "learning rate": 0.1,
            "batch size": 3
        }
        self.assertEqual(model.hyper_parameters["learning rate"], 0.1)
        self.assertEqual(model.hyper_parameters["batch size"], 3)

    def test_set_wrong_parameters(self):
        good_parameters = {"coef": 5, "intercept": 3}
        bad_parameters = {"BAD": 1}
        good_hyper_parameters = {"learning rate": 0.005, "batch size": 10}
        bad_hyper_parameters = {"WRONG KEY": 0.005, "batch size": 10}
        model = ConcreteModel(
            type="numerical",
            hyper_parameters=good_hyper_parameters,
            parameters=good_parameters
        )
        with self.assertRaises(AttributeError):
            model.hyper_parameters = bad_hyper_parameters

        with self.assertRaises(AttributeError):
            model.parameters = bad_parameters

class TestMultipleLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
        self.y = [
            [1],
            [2],
            [3],
        ]
        self.X_test = [
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ]

    def test_fit_and_predict(self):
        model = MultipleLinearRegression()
        model.fit(self.X, self.y)
        expected_prediction = [
            [4],
            [5],
            [6],
        ]
        prediction = model.predict(self.X_test)
        self.assertEqual(prediction, expected_prediction)

    def test_unset_predict(self):
        model = MultipleLinearRegression()
        with self.assertRaises(AssertionError) as context_manager:
            model.predict([1])
        exception = context_manager.exception
        self.assertEqual(str(exception), "Model is not fitted yet!")

    def test_to_and_from_artifact(self):
        model = MultipleLinearRegression()
        model.fit(self.X, self.y)
        artifact = model.to_artifact()
        new_model = MultipleLinearRegression.from_artifact(artifact)
        self.assertEqual(
            new_model.parameters["intercept"], model.parameters["intercept"]
        )

    def test_get_model(self):
        MultipleLinearRegression = get_model("MultipleLinearRegression")
        Model = MultipleLinearRegression()
        self.assertEqual(Model, MultipleLinearRegression)

class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.X = [
            [6, 1, 1],
            [2, 2, 4],
            [3, 2, 3],
        ]
        self.y = [
            [0],
            [1],
            [0],
        ]
        self.X_test = [
            [4, 4, 4],
            [5, 2, 5],
            [6, 6, 3],
        ]

def test_fit_and_predict(self):
    model = WrapLogisticRegression()
    model.fit(self.X, self.y)
    prediction = model.predict(self.X_test)
    isinstance(prediction, np.ndarray)

def test_unset_predict(self):
    model = WrapLogisticRegression()
    with self.assertRaises(AssertionError) as context_manager:
        model.predict([1])
    exception = context_manager.exception
    self.assertEqual(str(exception), "Model is not fitted yet!")

def test_to_and_from_artifact(self):
    model = WrapLogisticRegression()
    model.fit(self.X, self.y)
    artifact = model.to_artifact()
    new_model = WrapLogisticRegression.from_artifact(artifact)
    self.assertEqual(
        new_model.parameters["intercept"], model.parameters["intercept"]
    )

def test_get_model(self):
    WrapLogisticRegression = get_model("LogisticRegression")
    Model = WrapLogisticRegression()
    self.assertEqual(Model, WrapLogisticRegression)

# TODO: test setting hyperparameters


class TestElasticNet(unittest.TestCase):

    def setUp(self):
        self.X = [
            [6, 1, 1],
            [2, 2, 4],
            [3, 2, 3],
        ]
        self.y = [
            [1, 2],
            [2, 4],
            [3, 6],
        ]
        self.X_test = [
            [4, 4, 4],
            [5, 2, 5],
            [6, 6, 3],
        ]

def test_fit_and_predict(self):
    model = WrapLogisticRegression()
    model.fit(self.X, self.y)
    prediction = model.predict(self.X_test)
    isinstance(prediction, np.ndarray)

def test_unset_predict(self):
    model = WrapLogisticRegression()
    with self.assertRaises(AssertionError) as context_manager:
        model.predict([1])
    exception = context_manager.exception
    self.assertEqual(str(exception), "Model is not fitted yet!")

def test_to_and_from_artifact(self):
    model = WrapLogisticRegression()
    model.fit(self.X, self.y)
    artifact = model.to_artifact()
    new_model = WrapLogisticRegression.from_artifact(artifact)
    self.assertEqual(
        new_model.parameters["intercept"], model.parameters["intercept"]
    )

def test_get_model(self):
    WrapLogisticRegression = get_model("LogisticRegression")
    Model = WrapLogisticRegression()
    self.assertEqual(Model, WrapLogisticRegression)

# TODO: test setting hyperparameters


class TestKNearestNeighbors(unittest.TestCase):

    def setUp(self):
        self.X = [
            [6, 1, 1],
            [2, 2, 4],
            [3, 2, 3],
        ]
        self.y = [
            ["Boter"],
            ["Kaas"],
            ["Eiren"],
        ]
        self.X_test = [
            [4, 4, 4],
            [5, 2, 5],
            [6, 6, 3],
        ]

def test_fit_and_predict(self):
    model = WrapKNearestNeighbors()
    model.fit(self.X, self.y)
    prediction = model.predict(self.X_test)
    isinstance(prediction, np.ndarray)

def test_unset_predict(self):
    model = WrapKNearestNeighbors()
    with self.assertRaises(AssertionError) as context_manager:
        model.predict([1])
    exception = context_manager.exception
    self.assertEqual(str(exception), "Model is not fitted yet!")

def test_to_and_from_artifact(self):
    model = WrapKNearestNeighbors()
    model.fit(self.X, self.y)
    artifact = model.to_artifact()
    new_model = WrapKNearestNeighbors.from_artifact(artifact)
    self.assertEqual(
        new_model.parameters["classes"], model.parameters["classes"]
    )

def test_get_model(self):
    WrapLogisticRegression = get_model("KNearestNeighbor")
    Model = WrapKNearestNeighbors()
    self.assertEqual(Model, WrapKNearestNeighbors)

class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        self.X = [
            [6, 1, 1],
            [2, 2, 4],
            [3, 2, 3],
        ]
        self.y = [
            ["Boter"],
            ["Kaas"],
            ["Eiren"],
        ]
        self.X_test = [
            [4, 4, 4],
            [5, 2, 5],
            [6, 6, 3],
        ]

def test_fit_and_predict(self):
    model = WrapNaiveBayes()
    model.fit(self.X, self.y)
    prediction = model.predict(self.X_test)
    isinstance(prediction, np.ndarray)

def test_unset_predict(self):
    model = WrapNaiveBayes()
    with self.assertRaises(AssertionError) as context_manager:
        model.predict([1])
    exception = context_manager.exception
    self.assertEqual(str(exception), "Model is not fitted yet!")

def test_to_and_from_artifact(self):
    model = WrapNaiveBayes()
    model.fit(self.X, self.y)
    artifact = model.to_artifact()
    new_model = WrapNaiveBayes.from_artifact(artifact)
    self.assertEqual(
        new_model.parameters["classes"], model.parameters["classes"]
    )

def test_get_model(self):
    WrapLogisticRegression = get_model("KNearestNeighbor")
    Model = WrapNaiveBayes()
    self.assertEqual(Model, WrapNaiveBayes)


class TestRandomForest(unittest.TestCase):

    def setUp(self):
        self.X = [
            [6, 1, 1],
            [2, 2, 4],
            [3, 2, 3],
        ]
        self.y = [
            ["Boter"],
            ["Kaas"],
            ["Eiren"],
        ]
        self.X_test = [
            [4, 4, 4],
            [5, 2, 5],
            [6, 6, 3],
        ]

def test_fit_and_predict(self):
    model = WrapRandomForest()
    model.fit(self.X, self.y)
    prediction = model.predict(self.X_test)
    isinstance(prediction, np.ndarray)

def test_unset_predict(self):
    model = WrapRandomForest()
    with self.assertRaises(AssertionError) as context_manager:
        model.predict([1])
    exception = context_manager.exception
    self.assertEqual(str(exception), "Model is not fitted yet!")

def test_to_and_from_artifact(self):
    model = WrapRandomForest()
    model.fit(self.X, self.y)
    artifact = model.to_artifact()
    new_model = WrapRandomForest.from_artifact(artifact)
    self.assertEqual(
        new_model.parameters["classes"], model.parameters["classes"]
    )

def test_get_model(self):
    WrapLogisticRegression = get_model("KNearestNeighbor")
    Model = WrapRandomForest()
    self.assertEqual(Model, WrapRandomForest)


# TODO: test setting hyperparameters
