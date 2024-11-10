import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestCentroid

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
            type="nearest centroid",
            hyper_params=ParametersDict(hyper_params),  
            params=ParametersDict(params),
        )


        
        self._model = NearestCentroid(**hyper_params)
        if hyper_params.get("metric", None) is not None:
            self._model.metric = hyper_params["metric"]
        if hyper_params.get("shrink_threshold", None) is not None:
            self._model.shrink_threshold = hyper_params["shrink_threshold"]
        if hyper_params.get("priors", None) is not None:
            self._model.shrink_threshold = hyper_params["priors"]

        if params.get("centroids", None) is not None:
            self._model.centroids_ = params["centroids"]
        if params.get("classes", None) is not None:
            self._model.classes_ = params["classes"]

        #Im not adding all hyperparameters, this just seems kind of excessive
        


    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._params.update({
            "classes": self._model.classes_,
            "centroids": self._model.centroids_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._params["classes"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path = "./assets/models", version = "v0.00"):
        return super().to_artifact(
            name="nearest centroid model",
            asset_path= asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
