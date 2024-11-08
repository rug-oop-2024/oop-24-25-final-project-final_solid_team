from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.typing import ArrayLike

from autoop.core.ml.model.model import Model, ParametersDict

class MultipleLinearRegression(Model):
    def __init__(
            self,
            params: dict = ParametersDict({}),
            hyper_params: dict = ParametersDict({}),
        ) -> None:
        """_summary_

        Args:
            coef (np.ndarray): _description_
            intercept (float): _description_
        """
        super().__init__(
            type="multiple linear regression",
            hyper_params=ParametersDict(hyper_params),  # Superfluous
            params=ParametersDict(params),
        )
        self._model = LinearRegression(**hyper_params)
        if params.get("coef", None) is not None:
            self._model.coef_ = params["coef"]   
            # Only set intercept if coefficient is also set:
            if params.get("intercept", None) is not None:                                 
                self._model.intercept_ = params["intercept"]
    
        
    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._model.fit(X, y)
        self._params.update({
            "coef": self._model.coef_,
            "intercept": self._model.intercept_
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._params["coef"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)
        
    def to_artifact(self, asset_path = "./assets/models", version = "v0.00"):
        return super().to_artifact(
            name="multiple linear regression model",
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