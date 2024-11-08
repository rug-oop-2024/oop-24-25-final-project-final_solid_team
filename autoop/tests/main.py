import logging
import sys
import unittest

# from autoop.tests.test_database import TestDatabase
# from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_model import TestModel
# from autoop.tests.test_pipeline import TestPipeline

if __name__ == "__main__":
    logging.basicConfig(filename="test.log", filemode='w', level=logging.INFO)
    unittest.main()
