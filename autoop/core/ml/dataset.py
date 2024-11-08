import io

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """Artifact that represents a dataset."""
    
    def __init__(self, *args, **kwargs):
        """ Create a dataset object to store panda data frame into
        and byte-encoded csv-file.
        Args:
            name (str): Name of the artifact
            version (str): Version of the dataset. Default to "v0.00"
            tags (list[str]): Tags of the dataset. Defaults to None
            meta_data (str): Metadata.
            asset_path (str): Path to where the data is stored. Defaults to
                              None
            data (str): Binary data of the dataset. Defaults to None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ):
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)


# Remarks:
# read and save were already implemented

# Initial implementation:
# from abc import ABC, abstractmethod
# Maybe make this class abstract?

# Questions:
# Why does data have to be bytes. These artifacts are going to be
# decoded in storage anyway.
