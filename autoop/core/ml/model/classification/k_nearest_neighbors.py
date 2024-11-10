import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KNeighborsClassifier

from autoop.core.ml.model.model import Model, ParametersDict


class KNearestNeighbors(Model):
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
            type="k nearest neighbors",
            hyper_params=ParametersDict(hyper_params),  
            params=ParametersDict(params),
        )


        
        self._model = KNeighborsClassifier(**hyper_params)
        if hyper_params.get("k", None) is not None:
            self._model.n_neighbors = hyper_params["k"]
        if hyper_params.get("weights", None) is not None:
            self._model.weights = hyper_params["weights"]
        if hyper_params.get("leaf_size", None) is not None:
            self._model.leaf_size = hyper_params["leaf_size"]
        if hyper_params.get("algorithm", None) is not None:
            self._model.algorithm = hyper_params["algorithm"]
        if hyper_params.get("metric", None) is not None:
            self._model.metric = hyper_params["metric"]
        if hyper_params.get("p", None) is not None:
            self._model.p = hyper_params["p"]
        if hyper_params.get("metric_params", None) is not None:
            self._model.metric_params = hyper_params["metric_params"]

        if params.get("classes", None) is not None:
            self._model.classes_ = params["classes"]
        if params.get("effective_metric", None) is not None:
            self._model.effective_metric_ = params["effective_metric"]
        if params.get("effective_metric_params", None) is not None:
            self._model.effective_metric_params_ = params["effective_metric_params"]
        if params.get("n_features_in", None) is not None:
            self._model.n_features_in_ = params["n_features_in"]
        if params.get("feature_names_in", None) is not None:
            self._model.feature_names_in_ = params["feature_names_in"]
        if params.get("n_samples_fit", None) is not None:
            self._model.n_samples_fit_ = params["n_samples_fit"]
        if params.get("outputs_2d", None) is not None:
            self._model.outputs_2d_ = params["outputs_2d"]

        


    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._params.update({
            "classes": self._model.classes_,
            "effective_metric": self._model.effective_metric_,
            "effective_metric_params": self._model.effective_metric_params_,
            "n_features_in": self._model.n_features_in_,
            "feature_names_in": self._model.feature_names_in_,
            "n_samples_fit": self._model.n_samples_fit_,
            "outputs_2d": self._model.outputs_2d_,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._params["classes"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path = "./assets/models", version = "v0.00"):
        return super().to_artifact(
            name="k nearest neighbor model",
            asset_path= asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
