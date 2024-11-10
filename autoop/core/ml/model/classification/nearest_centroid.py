import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestCentroid

from autoop.core.ml.model.model import Model, ParametersDict


class NearestCentroid(Model):
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
        ) -> None:
        """_summary_

        Args:
                tba
          """
        super().__init__(
            type="nearest centroid",
            hyper_parameters=ParametersDict(hyper_parameters),  
            parameters=ParametersDict(parameters),
        )


        
        self._model = NearestCentroid(**hyper_parameters)
        if hyper_parameters.get("metric", None) is not None:
            self._model.metric = hyper_parameters["metric"]
        if hyper_parameters.get("shrink_threshold", None) is not None:
            self._model.shrink_threshold = hyper_parameters["shrink_threshold"]
        if hyper_parameters.get("priors", None) is not None:
            self._model.shrink_threshold = hyper_parameters["priors"]

        if parameters.get("centroids", None) is not None:
            self._model.centroids_ = parameters["centroids"]
        if parameters.get("classes", None) is not None:
            self._model.classes_ = parameters["classes"]

        #Im not adding all hyperparameters, this just seems kind of excessive
        


    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._parameters.update({
            "classes": self._model.classes_,
            "centroids": self._model.centroids_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._parameters["classes"] is not None, (
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
