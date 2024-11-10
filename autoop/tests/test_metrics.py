import unittest

import numpy as np

from autoop.core.ml.metric import Accuracy, MeanSquaredError, R_squared, Precision, Recall, Mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, accuracy_score



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

class TestR_squared(unittest.TestCase):
    def testValue(self):
        r_squared = R_squared()
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
        mabs = Mean_absolute_error()
        ground_truth: np.ndarray = np.array([2, 2, 3, 4])
        predictions: np.ndarray = np.array([3, 2, 4, 2])

        self.assertAlmostEqual(mabs(ground_truth, predictions), mean_absolute_error(ground_truth, predictions), 4)

