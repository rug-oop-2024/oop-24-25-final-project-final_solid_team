from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import numpy as np

from autoop.core.ml.artifact import Artifact


class Model(ABC):
    def __init__(self):
        _params: dict = dict()
        _hyper_params: dict = dict()
    
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

    
      # your code (attribute and methods) here
