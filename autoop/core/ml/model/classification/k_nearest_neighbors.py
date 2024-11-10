import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KNeighborsClassifier

from autoop.core.ml.model.model import Model, ParametersDict


class KNearestNeighbors(Model):
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
            type="k nearest neighbors",
            hyper_parameters=ParametersDict(hyper_parameters),  
            parameters=ParametersDict(parameters),
        )


        
        self._model = KNeighborsClassifier(**hyper_parameters)
        if hyper_parameters.get("k", None) is not None:
            self._model.n_neighbors = hyper_parameters["k"]
        if hyper_parameters.get("weights", None) is not None:
            self._model.weights = hyper_parameters["weights"]
        if hyper_parameters.get("leaf_size", None) is not None:
            self._model.leaf_size = hyper_parameters["leaf_size"]
        if hyper_parameters.get("algorithm", None) is not None:
            self._model.algorithm = hyper_parameters["algorithm"]
        if hyper_parameters.get("metric", None) is not None:
            self._model.metric = hyper_parameters["metric"]
        if hyper_parameters.get("p", None) is not None:
            self._model.p = hyper_parameters["p"]
        if hyper_parameters.get("metric_parameters", None) is not None:
            self._model.metric_parameters = hyper_parameters["metric_parameters"]

        if parameters.get("classes", None) is not None:
            self._model.classes_ = parameters["classes"]
        if parameters.get("effective_metric", None) is not None:
            self._model.effective_metric_ = parameters["effective_metric"]
        if parameters.get("effective_metric_parameters", None) is not None:
            self._model.effective_metric_parameters_ = parameters["effective_metric_parameters"]
        if parameters.get("n_features_in", None) is not None:
            self._model.n_features_in_ = parameters["n_features_in"]
        if parameters.get("feature_names_in", None) is not None:
            self._model.feature_names_in_ = parameters["feature_names_in"]
        if parameters.get("n_samples_fit", None) is not None:
            self._model.n_samples_fit_ = parameters["n_samples_fit"]
        if parameters.get("outputs_2d", None) is not None:
            self._model.outputs_2d_ = parameters["outputs_2d"]

        


    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._parameters.update({
            "classes": self._model.classes_,
            "effective_metric": self._model.effective_metric_,
            "effective_metric_parameters": self._model.effective_metric_parameters_,
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

    def to_artifact(self, asset_path = "./assets/models", version = "v0.00"):
        return super().to_artifact(
            name="k nearest neighbor model",
            asset_path= asset_path,
            version=version,
        )

    # From artifact does not have to be changed.
