import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import RadiusNeighborsClassifier

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model, ParametersDict


class WrapRadiusNeighor(Model):
    """Wrapper class for radius neighbors model of sklearn"""
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
    ) -> None:
        """
        Initialise wrapper for radius neighbors from sklearn.
        """
        super().__init__(
            type="classification",
            hyper_parameters=ParametersDict(hyper_parameters),
            parameters=ParametersDict(parameters),
        )
        #TODO Make auto adjusting hyperparam
        self._model = RadiusNeighborsClassifier(radius=20.0)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fits the radius neighbors sklearn model
        object to an array of input features and one of target features
        """
        self._model.fit(X, y)
        self._parameters.update({
            "classes": self._model.classes_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target features of input based on
        fit of radius neighbors model"""
        assert self._parameters["classes"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path: str = "./assets/models",
                    version: str = "v0.00") -> Artifact:
        """Convert model to artifact"""
        return super().to_artifact(
            name="radius neighbors model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
