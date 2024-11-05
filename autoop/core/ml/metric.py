from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from autoop.core.ml.dataset import Feature


METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    raise NotImplementedError("To be implemented.")


class Metric(ABC):
    """Base class for all metrics."""

    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number

    @abstractmethod
    def __call__(self, ground_truth: Feature, predictions: Feature) -> float:
        pass
        



class Accuracy(Metric):
    matches: int = 0
    def __call__(self, ground_truth: Feature, predictions: Feature) -> float:
        for index, item in enumerate(ground_truth.array()): #array method is just a placeholder
            if(predictions.array()[index] == item):
                matches+= 1
        return (matches / len(ground_truth.array()))
        

class Mean_squared_error(Metric):
    def __call__(self, ground_truth: Feature, predictions: Feature) -> float:
        difference_array: np.ndarray = predictions.array() - ground_truth.array()
        mean_sq_err:float = np.mean(difference_array**2)
        return mean_sq_err



# add here concrete implementations of the Metric class
