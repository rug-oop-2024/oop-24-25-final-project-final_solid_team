import logging
import sys
import unittest

from app.tests.test_system import TestAutoMLSystem
from autoop.tests.test_artifact import TestArtifact
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_model import TestModel
from autoop.tests.test_model import MultipleLinearRegression
from autoop.tests.test_model import TestElasticNet
from autoop.tests.test_model import TestKNearestNeighbors
from autoop.tests.test_model import TestLogisticRegression



from autoop.tests.test_pipeline import TestPipeline
from autoop.tests.test_storage import TestStorage

if __name__ == "__main__":
    logging.basicConfig(filename="test.log", filemode='w', level=logging.INFO)
    unittest.main()
