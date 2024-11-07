import unittest
import numpy as np

from autoop.core.ml.model import Model

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
