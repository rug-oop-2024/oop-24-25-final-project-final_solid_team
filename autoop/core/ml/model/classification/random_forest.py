import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier

from autoop.core.ml.model.model import Model, ParametersDict


class NearestCentroid(Model):
    def __init__(
            self,
            params: dict = ParametersDict({}),
            hyper_params: dict = ParametersDict({}),
        ) -> None:
        """_summary_

        Args:
                tba
          """
        super().__init__(
            type="random forest classifier",
            hyper_params=ParametersDict(hyper_params),  
            params=ParametersDict(params),
        )


        
        self._model = RandomForestClassifier(**hyper_params)
        


    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._params.update({
            "estimator": self._model.estimator_,
            "estimators": self._model.estimators_,
            "classes_": self._model.classes_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._params["estimator"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path = "./assets/models", version = "v0.00"):
        return super().to_artifact(
            name="random forest model",
            asset_path= asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
