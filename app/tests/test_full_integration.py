from __future__ import annotations
from typing import TYPE_CHECKING
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

if TYPE_CHECKING:
    import streamlit

class TestFullIntegration(unittest.TestCase):

    @patch("app.Welcome.st")
    @patch("app.Welcome.st.file_uploader")
    @patch("app.Welcome.st.write")
    def test_full_integration(
        self,
        mock_write, 
        file_uploader_mock, 
        mock_st: streamlit,
    ):
        with open("test_assets/iris.csv", mode="rb") as file:
            bytes = file.read()

        # Let st.file_uploader return an object which return bytes when
        # getvalue() is called
        file_object_mock = MagicMock()
        file_object_mock.getvalue.return_value = bytes
        file_object_mock.name = "iris.csv"
        file_uploader_mock.return_value = file_object_mock

        from app.pages.Datasets import main
        main()

        # mock_write.assert_called()
        mock_write.assert_any_call(f"Registered csv file iris.csv!")
    

if __name__ == "__main__":
    unittest.main()
