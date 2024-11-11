import tempfile
import unittest

import pandas as pd
from sklearn.datasets import fetch_openml

from app.core.pipline_handler import PipelineHandler
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types


class TestPipelineHandler(unittest.TestCase):
    def setUp(self):
        openml_object = fetch_openml("wine", version=1)
        df = pd.DataFrame(
            data=openml_object.data,
            names=openml_object.feature_names
        )
        self.dataset = Dataset.from_dataframe(
            df,
            name="test",
            asset_path=tempfile.mkdtemp(),
        )
        self.features = detect_feature_types(self.dataset)
        self.handler = PipelineHandler()

    def test_integration(self):
        pass