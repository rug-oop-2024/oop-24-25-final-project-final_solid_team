from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression

from autoop.core.ml.artifact import Artifact


# Inspired by Marco Zullich see: <https://www.rug.nl/staff/m.zullich/>
class ParametersDict(dict):
    def _get_keys_as_list(self) -> list:
        list_: list = list(self.keys())
        list_.sort()
        return list_

    def update(self, new_dict: dict) -> None:
        """Update the dictionary with new values.

        Args:
            new_dict (dict): Dictionary with the new values.

        Raises:
            AttributeError: Raises if the new dict does not have the same keys
                            as the old dict.
        """
        if not isinstance(new_dict, ParametersDict):
            new_dict = ParametersDict(new_dict)
        keys_same = self._get_keys_as_list() == new_dict._get_keys_as_list()
        if keys_same or len(self) == 0:
            super().update(new_dict)
        else:
            raise AttributeError(
                f"ParametersDict.update expects another ParametersDict with "
                f"the same keys. Found {self.keys()} vs {new_dict.keys()}"
            )


class Model(ABC):
    """Abstract base clase for any type of machine learning model."""
    def __init__(
            self,
            type: str,
            hyper_parameters: dict = ParametersDict({}),
            parameters: dict = ParametersDict({})
        ) -> None:
        self._type = type
        self._hyper_parameters = ParametersDict(hyper_parameters)
        self._parameters = ParametersDict(parameters)

    @staticmethod
    def from_artifact(artifact: Artifact) -> Model:
        """Load a model from an Artifact instance.

        Args:
            artifact (Artifact): Artifact in which the model is stored.

        Returns:
            Model: The loaded model.
        """
        return pickle.loads(artifact.read())

    def to_artifact(
            self,
            name: str = "baseclass model",
            asset_path: str = "./assets/models",
            **kwargs
        ) -> Artifact:
        """Get an artifact representation of the model.

        Args:
            name (str): Name of the artifact
            asset_path (str): Path to where the data is stored. Defaults to
            "./assets/models"
            version (str): Version of the artifact. Default to "v0.00"
            tags (list[str]): Tags of the artifact. Defaults to empy list
            meta_data (str): Metadata. Defaults to empty dictionary
        """
        return Artifact(
            name=name,
            type="model",
            data=pickle.dumps(self),
            asset_path=asset_path,
            **kwargs
        )

    @abstractmethod
    def fit(self, data: ArrayLike) -> None:
        pass

    @abstractmethod
    def predict(self, data: ArrayLike) -> np.ndarray:
        pass

    @property
    def type(self) -> str:
        return self._type

    @property
    def parameters(self) -> ParametersDict:
        """Getter for parameters."""
        return deepcopy(self._parameters)

    @property
    def hyper_parameters(self) -> ParametersDict:
        """Getter for hyperparameters"""
        return deepcopy(self._hyper_parameters)

    @parameters.setter
    def parameters(self, value: dict):
        self._parameters.update(value)

    @hyper_parameters.setter
    def hyper_parameters(self, hyperparameters: dict):
        """Setter for hyperparam ."""
        self._hyper_parameters.update(hyperparameters)

# TODO: Change (hyper)parameters into (hyper)parameters to be more consistent
# TODO Make update() more sophisticated:
# - Allow partial dict updates
