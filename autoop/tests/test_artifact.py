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


# TODO Test id attribute
