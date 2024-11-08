import base64

class Artifact:  # Original had Pydantic
    """Baseclass to store certain assets."""

    def __init__(
        self, *,  # Mandate usage of keywords
        type: str,
        name: str,
        version: str = "v0.00",
        tags: list[str] = [],
        metadata: dict[str, str] = dict(),
        asset_path: str,
        data: bytes,
    ) -> None:
        """Create an artifact object.

        Args:
            type (str): Type of the artifact.
            name (str): Name of the artifact
            version (str): Version of the artifact. Default to "v0.00"
            tags (list[str]): Tags of the artifact. Defaults to None
            meta_data (str): Metadata.
            asset_path (str): Path to where the data is stored. Defaults to
                              None
            data (str): Binary data of the artifact. Defaults to None
        """
        self._type = type
        self._name = name
        self._version = version
        self._tags = tags
        self._metadata = metadata
        self._asset_path = asset_path
        self._data = data

    def read(self) -> bytes:
        """Read the content of the data.

        Returns:
            The data stored in this artifact in binary.
        """
        return self.data

    def save(self, binary_string: bytes) -> bytes:
        """Save a binary string as data into this artifact.

        Args:
            binary_string: the data to be saved.
        """
        self.data = binary_string
        return binary_string
    
    @property
    def id(self) -> str:
        """Get the id of this artifact."""
        return {str(base64.b64encode(self.asset_path)) + self.version}
    
    @property
    def type(self) -> str:
        return self._type
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def tags(self) -> list[str]:
        return self._tags
    
    @property
    def metadata(self) -> dict[str, str]:
        return self._metadata
    
    @property
    def asset_path(self) -> str:
        return self._asset_path
    
    @property
    def data(self) -> bytes:
        return self._data

    @data.setter
    def data(self, value) -> None:
        if isinstance(value, bytes):
            self._data = value
        else:
            raise AttributeError(
                f"Invalid type of data. Data must be of type `bytes'. Got "
                f"type(value)"
            )


# TODO Make Artifact more strict: it must have everything but data
    # in particullary, it must have an asset_path because that is how a
    # artifact is referenced through
# TODO Figure out whether we really want utf-8 encoding
# TODO Understand base64.b64encode (now we just use .encode())
# TODO The Dataset.save implies that Artifact.save
    # function return a bytes object. Why is that?

# NOTE given file had:
# import base64
# Why is that?

# Reasonings:
# read and save must have these signatures based on how do are used in Dataset
# therefore it makes sense for the self._data to be of type bytes
# Artifact itself doesn't do much more then saving data. Other classes will
# have the task to do stuff with this data.
# the id property is based on how it is described in instructions.md

# Remarks:
# We internally use the getters of the attributes such we can later change
# how these atributes are stored


		