import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model, ParametersDict


class MultipleLinearRegression(Model):
    """Wrapper for the linear regression
    model from sklearn"""
    def __init__(
            self,
            parameters: dict = ParametersDict({}),
            hyper_parameters: dict = ParametersDict({}),
    ) -> None:
        """Initialise as numerical model
        """
        super().__init__(
            type="regression",
            hyper_parameters=ParametersDict(hyper_parameters),  # Superfluous
            parameters=ParametersDict(parameters),
        )
        self._model = LinearRegression(**hyper_parameters)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit multiple linear regression model
        to the data
        Args:
        X (ArrayLike): Input features
        Y (ArrayLike): Target features
        """

        self._model.fit(X, y)
        self._parameters.update({
            "coef": self._model.coef_,
            "intercept": self._model.intercept_
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target features based on input data
        and fitted model.
        Args:
        X (np.ndarray): Input features"""
        assert self._parameters["coef"] is not None, (
            "Model is not fitted yet!"
        )
        return self._model.predict(X)

    def to_artifact(self, asset_path: str = "./assets/models",
                    version: str = "v0.00") -> Artifact:
        """Convert model to artifact
        Args:
        asset_path (str): Path to asset folder
        version (str): version number
        Returns:
        Artifact"""
        return super().to_artifact(
            name="multiple linear regression model",
            asset_path=asset_path,
            version=version,
        )

    # From artifact does not have to be changed.

# Remarks
# Do we have to see one-hot encoded output feature as
# (number-of-categories x datapoints) output vector

# Probably better to bundle coef and intercept into one dict

# Could use kwargs in __init__ for compacter code but this is more semantic

# TODO Assert dimensions of array parameters
