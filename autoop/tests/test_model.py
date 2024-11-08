import unittest
import numpy as np

from autoop.core.ml.model.model import Model, ParametersDict
from autoop.core.ml.model.regression import MultipleLinearRegression


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
