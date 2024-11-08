import unittest

from autoop.core.ml.artifact import Artifact


class TestArtifact(unittest.TestCase):
    def setUp(self):
        self.init_args = {
            "type": "test", "name": "test",
            "asset_path": "/tmp/tmp","data": b"test bytes string"
        }

    def test_read_and_save(self):
        artifact = Artifact(**self.init_args)
        artifact.save(b"hello world")
        self.assertEqual(artifact.read(), b"hello world")

    def test_set_wrong_data(self):
        artifact = Artifact(**self.init_args)
        with self.assertRaises(AttributeError):
            artifact.data = "these are not bytes"


# Remarks:
# most of artifact 
# TODO Test id attribute
