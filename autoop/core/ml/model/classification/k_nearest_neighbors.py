import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KNeighborsClassifier

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model, ParametersDict


class WrapKNearestNeighbors(Model):
    """Wrapper for the K nearest neighbor model
    from sklearn. Child of model class
    """
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
    ) -> None:
        """Intiaialise k nearest neighbor model
        Initialise as classification type model"""

        super().__init__(
            type="classification",
            hyper_parameters=ParametersDict(hyper_parameters),
            parameters=ParametersDict(parameters),
        )

        self._model = KNeighborsClassifier(**hyper_parameters)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit k nearest neighbor model"""
        self._model.fit(X, y)
        self._parameters.update({
            "classes": self._model.classes_,
            "n_features_in": self._model.n_features_in_,
            "feature_names_in": self._model.feature_names_in_,
            "n_samples_fit": self._model.n_samples_fit_,
            "outputs_2d": self._model.outputs_2d_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target features of input using
          fitting k nearest neighbors model"""
        assert self._parameters["classes"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path: str = "./assets/models",
                    version: str = "v0.00") -> Artifact:
        """Convert k nearest neighbor model to artifact"""
        return super().to_artifact(
            name="k nearest neighbor model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
