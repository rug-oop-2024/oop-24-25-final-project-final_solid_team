import unittest

import pandas as pd
from sklearn.datasets import fetch_openml, load_iris

from app.core.system import ArtifactRegistry, AutoMLSystem
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.storage import LocalStorage


class TestAutoMLSystem(unittest.TestCase):
    def test_save_and_delete(self):
        automl = AutoMLSystem.get_instance()

        openml = fetch_openml("wine", version=1)

        iris = load_iris()
        iris_df = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names
        )
        iris_artifact = Dataset.from_dataframe(
            name="iris",
            data=iris_df,
            asset_path="datasets/iris"
        )
        automl.registry.register(iris_artifact)
        automl.registry.delete(iris_artifact.id)
