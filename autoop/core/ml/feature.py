from typing import Literal, Sized

import numpy as np
from numpy.typing import ArrayLike

from autoop.core.ml.dataset import Dataset


class Feature:
    def __init__(self, type: str, name: str, data: ArrayLike) -> None:
        """Create a feature.

        Args:
            type (str): Either "numerical" or "categorical".
            name (str): Description of this feature. E.g. age.
            data (ArrayLike): Data.
        """  # TODO  Improve data description.
        self._type = type
        self._name = name
        # Assert that data is an array, otherwise convert to array.
        self._data = np.asarray(data)

    @property
    def type(self) -> str:
        """Get the type."""
        return self._type
    
    @property
    def name(self) -> str:
        """Get the name."""
        return self._name 

    def __str__(self) -> str:
        """String representation of the object."""
        return (
            f"Type: {self._type}, Name: {self._name}\n"
            f"Data: {self._data}"
        )

    #Shoudn't this have a property to actually get the data?
    

# Reasonings:
# type and name have getters because they are accesed in the tests.
