from typing import Literal, Sized

import numpy as np

from autoop.core.ml.dataset import Dataset

OneZero = Literal[0, 1]
Categorical = tuple[OneZero]


class Feature:
    name: str

    def __init__(self, type: str, name: str, data: np.ndarray) -> None:
        """Create a feature.

        Args:
            type (str): Either "numerical" or "categorical".
            name (str): Description of this feature. E.g. age.
            data (np.ndarray): Data.
        """  # TODO  Improve data description.
        self._type = type
        self._name = name
        self._data = data


    @property
    def type(self):
        return self._type
    
    @property
    def name(self):
        return self._name

    def __str__(self):
        raise NotImplementedError("To be implemented.")
