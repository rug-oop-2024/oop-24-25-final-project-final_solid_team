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
        name: str,
        asset_path: str,
        version: str = "v0.00"
    ):
        super().__init__(
            type="model",
            name=name,
            asset_path=asset_path,
            version=version
        )

    def save(self, params: np.ndarray, hyperparams: np.ndarray):
        dict_ = {
            "params": params,
            "hyperparams": hyperparams
        }
        bytes = pickle.dumps(dict_)
        return super().save(bytes)

    def read(self) -> dict[str, np.ndarray]:
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
