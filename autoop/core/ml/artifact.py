import base64


class Artifact:  # Original had Pydantic
    """Baseclass to store certain assets."""

    _type: str
    _dictionary: dict

    def __init__(
        self,
        type: str | None = None,
        name: str | None = None,
        asset_path: str | None = None,
        data: bytes | None = None,
        version: str = "v0.00",
    ) -> None:
        """Initialize.

        Args:
            type (str): Type of the artifact.
            name (str): Name of the artifact
            asset_path (str): Path to where the data is stored
            data (str): Binary data
            version (str): Version of this artifact (default v0.00)
        """
        self._type = type,
        self._name = name,
        self._asset_path = asset_path,
        self._data = data
        self._version = version

    @property
    def asset_path(self) -> str:
        """Getter."""
        return self._asset_path

    @property
    def data(self) -> bytes:
        """Getter."""
        return self._data

    @property
    def id(self) -> dict[bytes, str]:
        """Getter."""
        return {base64.b64encode(self._asset_path), self._version}

    def read(self) -> bytes:
        """Read the content of the data.

        Returns:
            The data stored in this artifact in binary.
        """
        # TODO Do we need to assert that this
        # value exists?
        if self.data is not None:
            return self._data
        else:
            raise AttributeError("This artifact does not contain data (yet).")

    def save(self, binary_string: bytes) -> None:
        """Save a binary string as data into this artifact.

        Args:
            binary_string: the data to be saved.
        """
        self._data = binary_string
        return binary_string

# TODO Figure out whether we really want utf-8 encoding
# TODO Understand base64.b64encode
# TODO The Dataset.save implies that Artifact.save
# function return a bytes object. Why is that?

# Reasonings:
# read and save must have these signatures based on how do are used in Dataset
# therefore it makes sense for the self._data to be of type bytes
# Artifact itself doesn't do much more then saving data. Other classes will
# have the task to do stuff with this data.
# the id property is based on how it is described in instructions.md
