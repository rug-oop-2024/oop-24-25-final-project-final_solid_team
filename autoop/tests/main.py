import sys
import logging

import unittest

# from autoop.tests.test_database import TestDatabase
# from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures

# from autoop.tests.test_pipeline import TestPipeline

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(filename="test.log", level=logging.DEBUG)
    unittest.main()
