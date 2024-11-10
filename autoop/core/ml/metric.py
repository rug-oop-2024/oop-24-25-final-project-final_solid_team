from abc import ABC, abstractmethod

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
    # remember: metrics take ground truth and prediction
    # as input and return a real number

    @abstractmethod
    def __call__(self, ground_truth: Feature, predictions: Feature) -> float:
        pass

    def to_string(self) -> str:
        return ((str(self.__class__).split('.')[-1]).split("'"))[0]

    def __str__(self) -> str:
        # TODO Explain this line
        return ((str(self.__class__).split('.')[-1]).split("'"))[0]

# According to the pipeline it needs an .evaluate method instead of a call
# method


class Accuracy(Metric):
    """Class for accuracy metric"""

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        matches = 0
        """Accuracy __call__ function"""
        for index, item in enumerate(ground_truth):
            if (predictions[index] == item):
                matches += 1
        return (matches / len(ground_truth))


class MeanSquaredError(Metric):
    """Class for mean squared error metric"""
    def __init__(self):
        Metric.__init__(self)

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Mean squared error __call__ function"""
        difference_array: np.ndarray = predictions - ground_truth
        mean_sq_err: float = np.mean(difference_array**2)
        return mean_sq_err


class R_squared(Metric):
    """Class for R squared"""
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Mean squared error __call__ function"""
        ground_mean: float = np.mean(ground_truth)
        sum_squares: float = np.sum((ground_truth - ground_mean)**2)
        residual: float = np.sum((ground_truth - predictions)**2)

        return (1 - (residual / sum_squares))


class Precision(Metric):
    """Class for multi class precision metric"""
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Calculate the multi class precision"""
        precision_dict = {}
        for index, item in enumerate(predictions):
            if(item == ground_truth[index]):
                if item in precision_dict:
                    values = precision_dict.get(item)
                    values[0] += 1
                    precision_dict[item] = values
                else:
                    values = [1,0]
                    precision_dict[item] = values
            else:
                if item in precision_dict:
                    values = precision_dict.get(item)
                    values[1] += 1
                    precision_dict[item] = values
                else:
                    values = [0,1]
                    precision_dict[item] = values

        # perhaps generate error for empty array somehow
        mean_prc: int = 0
        for prc_value in precision_dict.values():
            mean_prc += (prc_value[0] / (prc_value[0] + prc_value[1]))
        truth_and_predictions = np.concatenate((ground_truth, predictions))
        # In case there are no predicted samples, we set prc to 0
        mean_prc = mean_prc / len(np.unique(truth_and_predictions))

        return mean_prc


class Recall(Metric):
    """Class for multi class recall metric"""
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Calculate mutli classs recall"""
        recall_dict = {}
        for index, item in enumerate(predictions):
            if(item == ground_truth[index]):
                if item in recall_dict:
                    values = recall_dict.get(item)
                    values[0] += 1
                    recall_dict[item] = values
                else:
                    values = [1,0]
                    recall_dict[item] = values

        for index, item in enumerate(ground_truth):
            if(item != predictions[index]):
                if item in recall_dict:
                    values = recall_dict.get(item)
                    values[1] += 1
                    recall_dict[item] = values
                else:
                    values = [0,1]
                    recall_dict[item] = values

        mean_rec: int = 0
        for rec_value in recall_dict.values():
            mean_rec += (rec_value[0] / (rec_value[0] + rec_value[1]))
        mean_rec = mean_rec / len(recall_dict)

        return mean_rec

class Mean_absolute_error(Metric):
    """Class for mean absolute error metric"""
    def __call__(self, ground_truth: np.ndarray,
                    predictions: np.ndarray) -> float:
        """Mean squared error __call__ function"""
        difference_array: np.ndarray = predictions - ground_truth
        absolute_diff: np.ndarray = np.abs(difference_array)
        return np.mean(absolute_diff)
