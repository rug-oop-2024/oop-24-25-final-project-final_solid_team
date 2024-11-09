import unittest

import numpy as np

from autoop.core.ml.model import MultipleLinearRegression, get_model
from autoop.core.ml.model.model import Model, ParametersDict


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
            hyper_params={"learning rate": 0.005, "batch size": 10},
            params={"slope": 0.5, "intercept": 2}
        )

        artifact = model.to_artifact(name="test artifact")
        recovered_model = ConcreteModel.from_artifact(artifact)

        self.assertEqual(model.type, recovered_model.type)
        self.assertEqual(model.params, recovered_model.params)
        self.assertEqual(model.hyper_params, recovered_model.hyper_params)

        # This test tested:
        # - initializer
        # - getters

    def test_set_hyper_params(self):
        model = ConcreteModel(
            type="numerical",
            hyper_params={"learning rate": 0.005, "batch size": 10},
            params={"slope": 0.5, "intercept": 2}
        )
        model.hyper_params = {
            "learning rate": 0.1,
            "batch size": 3
        }
        self.assertEqual(model.hyper_params["learning rate"], 0.1)
        self.assertEqual(model.hyper_params["batch size"], 3)

    def test_set_wrong_params(self):
        good_params = {"coef": 5, "intercept": 3}
        bad_params = {"BAD": 1}
        good_hyper_params = {"learning rate": 0.005, "batch size": 10}
        bad_hyper_params = {"WRONG KEY": 0.005, "batch size": 10}
        model = ConcreteModel(
            type="numerical",
            hyper_params=good_hyper_params,
            params=good_params
        )
        with self.assertRaises(AttributeError):
            model.hyper_params = bad_hyper_params

        with self.assertRaises(AttributeError):
            model.params = bad_params

class TestMultipleLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
        self.y = [
            [1, 2],
            [2, 4],
            [3, 6],
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
            [4, 8],
            [5, 10],
            [6, 12],
        ]
        prediction = model.predict(self.X_test)
        self.assertEqual(prediction, expected_prediction)

    def test_unset_predict(self):
        model = MultipleLinearRegression()
        with self.assertRaises(AssertionError) as context_manager:
            model.predict([1])
        exception = context_manager.exception
        self.assertEqual(str(exception), "Model is not fitted yet!")

    def test_setting_params(self):
        model = MultipleLinearRegression(params={"coef": 2, "intercept": 1})
        self.assertEqual(model.predict([[2]]), [5])

    def test_to_and_from_artifact(self):
        model = MultipleLinearRegression()
        model.fit(self.X, self.y)
        artifact = model.to_artifact()
        new_model = MultipleLinearRegression.from_artifact(artifact)
        self.assertEqual(
            new_model.params["intercept"], model.params["intercept"]
        )

    def test_get_model(self):
        MultipleLinearRegression = get_model("MultipleLinearRegression")
        Model = MultipleLinearRegression()
        self.assertEqual(Model, MultipleLinearRegression)


# TODO: test setting hyperparams
