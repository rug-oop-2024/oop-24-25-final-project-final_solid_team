from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Base class for all metrics."""

    # your code here
    # remember: metrics take ground truth and prediction
    # as input and return a real number

    @abstractmethod
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Take ground truth as np ndarray
        and predictions  as np ndarray,
        return metric value as float"""
        pass

    def to_string(self) -> str:
        """Return class string"""
        try:
            return ((str(self.__class__).split('.')[-1]).split("'"))[0]
        except Exception as e:
            print(
                "Unexpted error occured please contact the author:\n"
                "<g.j.wiersma@student.rug.nl\n"
                f"Exception info:\n{e}"
            )
            return "Undefinied metric"  # Attempting to not break the program

# According to the pipeline it needs an .evaluate method instead of a call
# method


class Accuracy(Metric):
    """Class for accuracy metric"""

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Accuracy __call__ function"""
        matches = 0
        for index, item in enumerate(ground_truth):
            equiv = False
            # Deal with case where arrays dont have one hot
            if (isinstance(item, np.ndarray)):
                equiv = ((item == (predictions[index])).all())
            else:
                equiv = (item == predictions[index])

            if (equiv):
                matches += 1
        return (matches / len(predictions))


class MeanSquaredError(Metric):
    """Class for mean squared error metric"""

    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Mean squared error __call__ function"""
        difference_array: np.ndarray = predictions - ground_truth
        mean_sq_err: float = np.mean(difference_array**2)
        return mean_sq_err


class RSquared(Metric):
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
            equiv = False
            # Deal with case where arrays dont have one hot
            if (isinstance(item, np.ndarray)):
                equiv = ((item == ground_truth[index]).all())
                key_item = tuple(item)
            else:
                equiv = (item == ground_truth[index])
                key_item = item

            if (equiv):
                if key_item in precision_dict:
                    values = precision_dict.get(key_item)
                    values[0] += 1
                    precision_dict[key_item] = values
                else:
                    values = [1, 0]
                    precision_dict[key_item] = values
            else:
                if key_item in precision_dict:
                    values = precision_dict.get(key_item)
                    values[1] += 1
                    precision_dict[key_item] = values
                else:
                    values = [0, 1]
                    precision_dict[key_item] = values

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
            equiv = False
            # Deal with case where arrays dont have one hot
            if (isinstance(item, np.ndarray)):
                equiv = ((item == (ground_truth[index])).all())
                key_item = tuple(item)
            else:
                equiv = (item == ground_truth[index])
                key_item = item

            if (equiv):
                if key_item in recall_dict:
                    values = recall_dict.get(key_item)
                    values[0] += 1
                    recall_dict[key_item] = values
                else:
                    values = [1, 0]
                    recall_dict[key_item] = values

        for index, item in enumerate(ground_truth):

            equiv = False
            # Deal with case where arrays dont have one hot
            if (isinstance(item, np.ndarray)):
                equiv = ((item == (predictions[index])).all())
                key_item = tuple(item)
            else:
                equiv = (item == predictions[index])
                key_item = item

            if (not equiv):
                if item in recall_dict:
                    values = recall_dict.get(key_item)
                    values[1] += 1
                    recall_dict[key_item] = values
                else:
                    values = [0, 1]
                    recall_dict[key_item] = values

        mean_rec: int = 0
        for rec_value in recall_dict.values():
            mean_rec += (rec_value[0] / (rec_value[0] + rec_value[1]))
        mean_rec = mean_rec / len(recall_dict)

        return mean_rec


class MeanAbsoluteError(Metric):
    """Class for mean absolute error metric"""
    def __call__(self, ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """Mean squared error __call__ function"""
        difference_array: np.ndarray = predictions - ground_truth
        absolute_diff: np.ndarray = np.abs(difference_array)
        return np.mean(absolute_diff)


CLASSIFICATION_METRICS = {
    "Precison": Precision,
    "Accuracy": Accuracy,
    "Recall": Recall,
}

REGRESSION_METRICS = {
    "R Squared": RSquared,
    "Mean Squared Error": MeanSquaredError,
    "Mean Abosolute Error": MeanAbsoluteError
}

# Time hurry
METRICS = {
    "Precison": Precision,
    "Accuracy": Accuracy,
    "Recall": Recall,
    "R Squared": RSquared,
    "Mean Squared Error": MeanSquaredError,
    "Mean Abosolute Error": MeanAbsoluteError
}
