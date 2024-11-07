import unittest
import numpy as np

from autoop.core.ml.model import Model

class ConcreteModel(Model):
    def fit(self, data) -> None:
        pass

    def predict(self, data) -> np.ndarray:
        pass

class TestModel(unittest.TestCase):
    def test_save_and_read(self):
        model = ConcreteModel(name="test model", asset_path="/tmp")
        params = np.array([[1, 2, 3.2], [4, 5, 6], [7, 8, 9]])
        hyperparams = np.array([0.1, 4])
        model.save(params, hyperparams)
        dict_ = model.read()
        self.assertTrue((dict_["params"] == params).all())
        self.assertTrue((dict_["hyperparams"] == hyperparams).all())
