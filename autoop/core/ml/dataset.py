from __future__ import annotations
import io

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """Artifact that represents a dataset."""
    
    def __init__(self, **kwargs):
        """ Create a dataset object to store panda data frame into
        and byte-encoded csv-file.
        Args:
            name (str): Name of the artifact
            asset_path (str): Path to where the data is stored. Defaults to
                              None
            data (str): Binary data of the dataset. Defaults to None
            version (str): Version of the dataset. Defaults to "v0.00".
            tags (list[str]): Tags of the dataset. Defaults to None
            meta_data (str): Metadata.
        """
        super().__init__(type="dataset", **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, **kwargs
    ) -> Dataset:
        """Returns a Dataset instance from a panda dataframe.

        Args:
            data (pd.DataFrame): Data from to be stored.
            name (str): Name of the artifact
            version (str): Version of the dataset. Defaults to "v0.00".
            tags (list[str]): Tags of the dataset. Defaults to None
            meta_data (str): Metadata.
            asset_path (str): Path to where the data is stored. Defaults to
                              None

        Returns:
            Dataset: The created Dataset instance.
        """
        return Dataset(
            data=data.to_csv(index=False).encode(), **kwargs
        )

    def read(self) -> pd.DataFrame:
        """Read the data from the dataset.

        Returns:
            pd.DataFrame: The panda datafame that was stored in this dataset.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save a panda dataframe into this dataset. Also return the encoded
        version of the dataframe.

        Args:
            data (pd.DataFrame): Panda dataframe to be stored.

        Returns:
            bytes: The bytes that where stored.
        """
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
