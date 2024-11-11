import unittest

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from autoop.core.ml.metric import (
    Accuracy,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
    RSquared,
)
from autoop.core.ml.model.model import Model, ParametersDict


class TestAccuracy(unittest.TestCase):
    def testValue(self):
        accuracy = Accuracy()
        ground_truth: np.ndarray = np.array([6, 2, 3, 1])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(accuracy(np.asarray(ground_truth), predictions), accuracy_score(ground_truth, predictions), 4)

class TestMeanSquaredError(unittest.TestCase):
    def testValue(self):
        mse = MeanSquaredError()
        ground_truth: np.ndarray = np.array([6, 2, 3, 1])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(mse(ground_truth, predictions), mean_squared_error(ground_truth, predictions), 4)

class TestRsquared(unittest.TestCase):
    def testValue(self):
        r_squared = RSquared()
        ground_truth: np.ndarray = np.array([6, 2, 3, 1])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(r_squared(ground_truth, predictions), r2_score(ground_truth, predictions), 4)

class TestPrecision(unittest.TestCase):
    def testValue(self):
        precision = Precision()
        ground_truth: np.ndarray = np.array([2, 2, 3, 4])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(precision(ground_truth, predictions), precision_score(ground_truth, predictions, average='macro'), 4)

class TestRecall(unittest.TestCase):
    def testValue(self):
        precision = Recall()
        ground_truth: np.ndarray = np.array([2, 2, 3, 4])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(precision(ground_truth, predictions), recall_score(ground_truth, predictions, average='macro'), 4)

class TestMeanAbsoluteError(unittest.TestCase):
    def testValue(self):
        mabs = MeanAbsoluteError()
        ground_truth: np.ndarray = np.array([2, 2, 3, 4])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(mabs(ground_truth, predictions), mean_absolute_error(ground_truth, predictions), 4)

