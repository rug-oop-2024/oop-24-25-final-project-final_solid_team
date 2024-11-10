import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestCentroid

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model, ParametersDict


class WrapNearestCentroid(Model):
    """Wrapper class for nearest centroid of sklearn"""
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
    ) -> None:
        """
        Initialise wrapper for nearest centroid from sklearn.
        """
        super().__init__(
            type="classification",
            hyper_parameters=ParametersDict(hyper_parameters),
            parameters=ParametersDict(parameters),
        )

        self._model = NearestCentroid(**hyper_parameters)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fits the nearestest centroid sklearn model
        object to an array of input features and one of target features
        """
        self._model.fit(X, y)
        self._parameters.update({
            "classes": self._model.classes_,
            "centroids": self._model.centroids_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target features of input based on
        fit of nearest centroid model"""
        assert self._parameters["classes"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path: str = "./assets/models",
                    version: str = "v0.00") -> Artifact:
        """Convert model to artifact"""
        return super().to_artifact(
            name="nearest centroid model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
