class Feature:
    """Represents a feature in a dataset."""
    def __init__(self, type: str, name: str) -> None:
        """Create a feature.

        Args:
            type (str): Either "numerical" or "categorical".
            name (str): Description of this feature. E.g. age.
        """
        self._type = type
        self._name = name

    @property
    def type(self) -> str:
        """Get the type."""
        return self._type

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    def __str__(self) -> str:
        """String representation of the object."""
        return (
            f"Type: {self._type}, Name: {self._name}\n"
        )


# Reason:
# type and name have getters because they are accesed in the tests.

# NOTE:
# The given implementation had the line:
# from typing import Literal, Sized
# on top? Why was that?
