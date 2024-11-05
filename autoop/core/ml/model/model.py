from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal, Any

import numpy as np
import pickle

from autoop.core.ml.artifact import Artifact

#HACK
ParamType = np.ndarray
HyperParamType = Any


class Model(Artifact):
    def __init__(
        self,
        type: str | None = None,
        name: str | None = None,
        asset_path: str | None = None,
        version: str = "v0.00",
    ):
        super().__init__(
            self,
            type=type,
            name=name,
            asset_path=asset_path,
            version=version,
            data=None,  # Model should not be initialised with parameters.
        )

    @abstractmethod
    def fit(self, data) -> None:
        pass

    @abstractmethod
    def predict(self, data) -> np.ndarray:
        pass

    @property
    def params(self) -> str:
        """Getter for params."""
        return deepcopy(self._params)

    @property
    def hyper_params(self) -> dict:
        """Getter for hyperparams"""
        return deepcopy(self._hyper_params)

    @hyper_params.setter
    def hyper_params(self, hypers: dict) -> str:
        """Setter for hyperparam ."""
        self._hyper_params = hypers
        return self._type


# WORKING ON RIGHT NOW:
# - Read and save should be able to read and save from a file. 
# - Data should be stored in a pickled dict {"params": ..., "hyperparams": ...}
