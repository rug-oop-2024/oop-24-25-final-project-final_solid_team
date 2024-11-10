import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from autoop.core.ml.model.model import Model, ParametersDict
from autoop.core.ml.artifact import Artifact


class WrapRandomForest(Model):
    """Wrapper for the sklearn random forest class
    child of the general model class"""
    def __init__(
            self,
            params: dict = ParametersDict({}),
            hyper_params: dict = ParametersDict({}),
    ) -> None:
        """
        Wrapper for the sklearn random forest class
        initialize as classification type model.
          """
        super().__init__(
            type="classification",
            hyper_params=ParametersDict(hyper_params),
            params=ParametersDict(params),
        )

        self._model = RandomForestClassifier(**hyper_params)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit random forest model to set
        of input and target features"""
        self._model.fit(X, y)
        self._params.update({
            "estimator": self._model.estimator_,
            "estimators": self._model.estimators_,
            "classes_": self._model.classes_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target features based on
        fitted random forest model"""
        assert self._params["estimator"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path: str = "./assets/models",
                    version: str = "v0.00") -> Artifact:
        """Covert random forest model to artifact"""
        return super().to_artifact(
            name="random forest model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
