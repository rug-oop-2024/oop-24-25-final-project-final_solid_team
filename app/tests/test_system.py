import unittest

from app.core.system import ArtifactRegistry
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.storage import LocalStorage


class TestArtifactRegistry(unittest.TestCase):
    def test_register(self):
        storage = LocalStorage()
        database = Database(storage)
        registry = ArtifactRegistry(
            storage=storage,
            database=database
        )

        dataset = Dataset(
            name="test dataset",
            asset_path="/tmp/tmp",
            data=b"hello world",
        )

        registry.register(dataset)

