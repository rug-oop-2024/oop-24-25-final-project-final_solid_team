from pydantic import BaseModel
import base64


class Artifact(BaseModel):
    """Baseclass to store certain assets."""
    _type: str
    _dictionary: dict

    def __init__(
        self,
        type: str,
        name: str,
        asset_path: str | None = None,
        data: bytes | None = None,
        version: str = "v0.00",
    ) -> None:
        """
        Args:
            type (str): Type of the artifact.
            name (str): Name of the artifact
            asset_path (str): Path to where the data is stored
            data (str): Binary data
        """
        self._type = type
        self._dictionary = {
            "name": name,
            "asset_path": asset_path,
            "version": version,
            "data": data,
        }

    def read(self) -> bytes:
        """Read the content of the data

        Returns:
            The data stored in this artifact in binary.
        """
                                       # TODO Do we need to assert that this
                                       # value exists?
        return self._dictionary["data"]

    def save(self, binary_string) -> None:
        """Save a binary string as data into this artifact.
        Args:
            binary_string: the data to be saved.
        """
        self._dictionary["data"] = binary_string

# TODO Figure out whether we really want utf-8 encoding
# TODO Maybe get rid of the dictionarry
