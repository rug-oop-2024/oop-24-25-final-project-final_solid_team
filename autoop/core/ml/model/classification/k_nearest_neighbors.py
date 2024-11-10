import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KNeighborsClassifier

from autoop.core.ml.model.model import Model, ParametersDict


class WrapKNearestNeighbors(Model):
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
        ) -> None:

        super().__init__(
            type="classification",
            hyper_parameters=ParametersDict(hyper_parameters),
            parameters=ParametersDict(parameters),
        )

        self._model = KNeighborsClassifier(**hyper_parameters)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._parameters.update({
            "classes": self._model.classes_,
            "n_features_in": self._model.n_features_in_,
            "feature_names_in": self._model.feature_names_in_,
            "n_samples_fit": self._model.n_samples_fit_,
            "outputs_2d": self._model.outputs_2d_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._parameters["classes"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path="./assets/models", version="v0.00"):
        return super().to_artifact(
            name="k nearest neighbor model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
