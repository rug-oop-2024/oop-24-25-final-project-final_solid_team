from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from autoop.core.ml.artifact import Artifact


# Author: Marco Zullich
class ParametersDict(dict):
    def _get_keys_as_list(self) -> list:
        list_: list = list(self.keys())
        list_.sort()
        return list_

    def update(self, new_dict: dict) -> None:
        if not isinstance(new_dict, ParametersDict):
            new_dict = ParametersDict(new_dict)
        if (self._get_keys_as_list() == new_dict._get_keys_as_list()
            or len(self) == 0
        ):
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
            hyper_params: dict = ParametersDict({}),
            params: dict = ParametersDict({})
        ) -> None:
        self._type = type
        self._hyper_params = ParametersDict(hyper_params)
        self._params = ParametersDict(params)

    @staticmethod
    def from_artifact(artifact: Artifact) -> Model:
        return pickle.loads(artifact.read())

    def to_artifact(
            self,
            name: str,
            asset_path: str = "./assets/models",
            version: str = "v0.00",
        ) -> Artifact:
        return Artifact(
            name=name,
            type="model",
            data=pickle.dumps(self),
            asset_path=asset_path,
            version=version,
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
    def params(self) -> ParametersDict:
        """Getter for params."""
        return deepcopy(self._params)

    @property
    def hyper_params(self) -> ParametersDict:
        """Getter for hyperparams"""
        return deepcopy(self._hyper_params)

    @params.setter
    def params(self, value: dict):
        self._params.update(value)

    @hyper_params.setter
    def hyper_params(self, hyperparams: dict):
        """Setter for hyperparam ."""
        self._hyper_params.update(hyperparams)

# TODO Make update() more sophisticated:
# - Allow partial dict updates
