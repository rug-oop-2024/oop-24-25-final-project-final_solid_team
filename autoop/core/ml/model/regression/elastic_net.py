import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import ElasticNet

from autoop.core.ml.model.model import Model, ParametersDict
from autoop.core.ml.artifact import Artifact


class WrapElasticNet(Model):
    """Wrapper for the sklearn elasticnet class
    child of the generalized mode class"""
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
    ) -> None:
        """
        Initalize the elasticnet wrapper
        as regression type model.
        """
        super().__init__(
            type="regression",
            hyper_parameters=ParametersDict(hyper_parameters),
            parameters=ParametersDict(parameters),
        )
        self._model = ElasticNet(**hyper_parameters)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit elasticnet model to input features"""
        self._model.fit(X, y)
        self._parameters.update({
            "coef": self._model.coef_,
            "intercept": self._model.intercept_
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target features based on fitted
        elasticnet model"""
        assert self._parameters["coef"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path: str = "./assets/models",
                    version: str = "v0.00") -> Artifact:
        """Convert model to artifact"""
        return super().to_artifact(
            name="elastic net model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.

# Remarks
# Do we have to see one-hot encoded output feature as
# (number-of-categories x datapoints) output vector

# Probably better to bundle coef and intercept into one dict

# Could use kwargs in __init__ for compacter code but this is more semantic

# TODO Assert dimensions of array parameters
