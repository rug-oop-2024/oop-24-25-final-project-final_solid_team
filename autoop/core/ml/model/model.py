from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal, Any

import numpy as np
import pickle

from autoop.core.ml.artifact import Artifact

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import numpy as np

from autoop.core.ml.artifact import Artifact


# Author: Marco Zullich
class ParametersDict(dict):
    def _get_keys_as_list(self) -> list:
        return list(self.keys()).sort()
    
    def update(self, new_dict: ParametersDict) -> None:
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
    def __init__(
            self,
            type: str,
            hyper_params: ParametersDict = ParametersDict({}),
            params: ParametersDict = ParametersDict({})
        ) -> None:
        self._type = type
        self._hyper_params = hyper_params
        self._params = params

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
    def fit(self, data) -> None:
        pass

    @abstractmethod
    def predict(self, data) -> np.ndarray:
        pass

    @property
    def type(self) -> str:
        return self._type

    @property
    def params(self) -> str:
        """Getter for params."""
        return deepcopy(self._params)

    @property
    def hyper_params(self) -> dict:
        """Getter for hyperparams"""
        return deepcopy(self._hyper_params)

    @hyper_params.setter
    def hyper_params(self, hyperparams: dict) -> str:
        """Setter for hyperparam ."""
        self._hyper_params.update(hyperparams)
