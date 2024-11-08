from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from autoop.core.ml.feature import Feature

METRICS = [
    "MeanSquareError",
    "Accuracy",
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


# Needs a __str__ method to display the name of the matric
# According to the pipeline it needs an .evaluate method instead of a call
# method






# add here concrete implementations of the Metric class

class Accuracy(Metric):
    """Class for accuracy metric"""

    def __call__(self, ground_truth: Feature, predictions: Feature) -> float:
        matches  = 0
        """Accuracy __call__ function"""
        for index, item in enumerate(ground_truth.data()):
            if (predictions.data()[index] == item):
                matches += 1
        return (matches / len(ground_truth.data()))


class MeanSquaredError(Metric):
    """Class for mean squared error metric"""
    def __call__(self, ground_truth: Feature, predictions: Feature) -> float:
        """Mean squared error __call__ function"""
        difference_array: np.ndarray = predictions.data() - ground_truth.data()
        mean_sq_err:float = np.mean(difference_array**2)
        return mean_sq_err


