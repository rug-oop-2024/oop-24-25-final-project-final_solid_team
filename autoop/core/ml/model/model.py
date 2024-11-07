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
    """Artifact that represents the a model and its parameters."""
    def __init__(
            self,
            name: str,
            asset_path: str,
            version: str = "v0.00"
        ):
        """Initialise a Model artifact.

        Args:
            name (str): Name of the artifact
            asset_path (str): Location where the data is allowed to be stored
            version (str, optional): Version. Defaults to "v0.00".
        """
        super().__init__(
            type="model",
            name=name,
            asset_path=asset_path,
            version=version
        )

    def save(self, params: np.ndarray, hyperparams: np.ndarray) -> bytes:
        """
        Save parameters (usually the weights of the model) and the
        hyperparameters (the knobs for traing) into the artifact.

        Args:
            params (np.ndarray): Parameters (weights) of the model.
            hyperparams (np.ndarray): Hyperparameters of the model.

        Returns:
            bytes: bytes representation of the data that is stored.
        """
        dict_ = {
            "params": params,
            "hyperparams": hyperparams
        }
        bytes = pickle.dumps(dict_)
        return super().save(bytes)

    def read(self) -> dict[str, np.ndarray]:
        """Read the data of the artifact. 

        Returns:
            dict[str, np.ndarray]: Dictionarry with params and hyperparams key
        """
        bytes = super().read()
        dict_ = pickle.loads(bytes)
        return dict_

    @abstractmethod
    def fit(self, data) -> None:
        pass

    @abstractmethod
    def predict(self, data) -> np.ndarray:
        pass


# Reasoning
# storing the parameters in a encoding dictionary. Artifacts need to will be
# json serialized and encoded and it's therefore better for all the attributes
# of an artifact to be strings or bytes
