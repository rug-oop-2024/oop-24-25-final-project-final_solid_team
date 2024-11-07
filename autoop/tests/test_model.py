import unittest
import numpy as np

from autoop.core.ml.model.model import Model

class ConcreteModel(Model):
    def fit(self, data) -> None:
        pass

    def predict(self, data) -> np.ndarray:
        pass

class TestModel(unittest.TestCase):
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
        good_hyper_params = {"learning rate": 0.005, "batch size": 10}
        bad_hyper_params = {"WRONG KEY": 0.005, "batch size": 10}
        model = ConcreteModel(
            type="numerical",
            hyper_params=good_hyper_params,
            params={"slope": 0.5, "intercept": 2}
        )
        with self.assertRaises(AttributeError) as exception:
            model.hyper_params = bad_hyper_params
