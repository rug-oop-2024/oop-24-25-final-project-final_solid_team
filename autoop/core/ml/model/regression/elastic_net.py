import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import ElasticNet

from autoop.core.ml.model.model import Model, ParametersDict


class ElasticNet(Model):
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
        ) -> None:
        """_summary_

        Args:
            coef (np.ndarray): _description_
            intercept (float): _description_
        """
        super().__init__(
            type="elastic net",
            hyper_parameters=ParametersDict(hyper_parameters),  
            parameters=ParametersDict(parameters),
        )
        self._model = ElasticNet(**hyper_parameters)
        if parameters.get("coef", None) is not None:
            self._model.coef_ = parameters["coef"]
            # Only set intercept if coefficient is also set:
            if parameters.get("intercept", None) is not None:
                self._model.intercept_ = parameters["intercept"]
        if hyper_parameters.get("alpha", None) is not None:
            self._model.alpha = hyper_parameters["alpha"]
        if hyper_parameters.get("l1_ratio", None) is not None:
            self._model.l1_ratio = hyper_parameters["l1_ratio"]
        if hyper_parameters.get("fit_intercept", None) is not None:
            self._model.fit_intercept = hyper_parameters["fit_intercept"]
        

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._parameters.update({
            "coef": self._model.coef_,
            "intercept": self._model.intercept_
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._parameters["coef"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path = "./assets/models", version = "v0.00"):
        return super().to_artifact(
            name="elastic net model",
            asset_path= asset_path,
            version=version,
        )

    # From artifact does not have to be changed.

# Remarks
# Do we have to see one-hot encoded output feature as
# (number-of-categories x datapoints) output vector

# Probably better to bundle coef and intercept into one dict

# Could use kwargs in __init__ for compacter code but this is more semantic

# TODO Assert dimensions of array parameters