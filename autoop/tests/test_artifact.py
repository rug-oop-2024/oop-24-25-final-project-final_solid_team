import unittest
from autoop.core.ml.artifact import Artifact

class TestArtifact(unittest.TestCase):
    def test_read_save(self):
        artifact = Artifact()
        artifact.save(b"hello world")
        self.assertEqual(artifact.read(), b"hello world")

    def test_read_exception(self):
        artifact = Artifact()
        with self.assertRaises(AttributeError):
            artifact.read()


# TODO Test id attribute
