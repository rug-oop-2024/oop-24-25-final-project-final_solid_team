class Artifact:  # Original had Pydantic
    """Baseclass to store certain assets."""

    def __init__(
        self,
        type: str | None = None,
        name: str | None = None,
        asset_path: str | None = None,
        version: str = "v0.00",
        data: bytes | None = None,
    ) -> None:
        """Initialize.

        Args:
            type (str): Type of the artifact.
            name (str): Name of the artifact
            asset_path (str): Path to where the data is stored
            data (str): Binary data
            version (str): Version of this artifact (default v0.00)
        """
        self._type = type
        self._name = name
        self._asset_path = asset_path
        self._data = data
        self._version = version

    @property
    def asset_path(self) -> str:
        """Getter."""
        if self._asset_path is not None:
            return self._asset_path
        else:
            raise AttributeError("asset_path is not set.")

    @property
    def data(self) -> bytes:
        """Getter."""
        if self._data is not None:
            return self._data
        else:
            raise AttributeError("data is not set.")

    @property
    def id(self) -> dict[bytes, str]:
        """Get the id of this artifact."""
        return {self.asset_path.encode(): self._version}

    def read(self) -> bytes:
        """Read the content of the data.

        Returns:
            The data stored in this artifact in binary.
        """
        if self._data:
            return self._data
        if self._asset_path:
            return self._get_data(self._asset_path)
        
        raise AttributeError("This attribute is not yet set.")

    def save(self, binary_string: bytes) -> bytes:
        """Save a binary string as data into this artifact.

        Args:
            binary_string: the data to be saved.
        """
        self._data = binary_string
        self._save_data(binary_string)
        return binary_string
    
# WORKING ON RIGHT NOW:
# - Make read and save such that is reads and saves from a data file in 
# assets/
# - Make Artifact more strict: it must have everything but data
# in particullary, it must have an asset_path because that is how a artifact
# is referenced through

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
