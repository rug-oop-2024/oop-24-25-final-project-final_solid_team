
from typing import Literal, Sized
import numpy as np

from autoop.core.ml.dataset import Dataset

OneZero = Literal[0, 1]
Categorical = tuple[OneZero]

class Feature:
    # attributes here
    _type: str

    def __init__(self, data: Sized, type: str):
        self._type = type
        self._data = data

    @property
    def type(self):
        return self._type

    def __str__(self):
        raise NotImplementedError("To be implemented.")
