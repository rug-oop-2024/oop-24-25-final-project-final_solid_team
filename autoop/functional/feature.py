import logging
from typing import Any, List

import pandas as pd

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

logger = logging.getLogger(__name__)
rejection_logger = logging.getLogger("rejection_92165")


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    assert isinstance(dataset, Dataset), (
        f"Excpected a Dataset for dataset. Got {type(dataset)}"
    )
    df = dataset.read()

    features = []
    for column_name in df:
        if df[column_name].hasnans:
            continue  # Do not add features with NaNs to features
        if _is_numerical(df[column_name]):
            feature = Feature(
                type="numerical",
                name=str(column_name),
            )
            features.append(feature)
        # elif such this one does not need to be checked if numerical
        elif _is_categorical(df[column_name]):
            feature = Feature(
                type="categorical",
                name=str(column_name),
            )
            features.append(feature)
        else:
            rejection_logger.info(
                f"Column {column_name} with type {df[column_name].dtype} is "
                "rejected"
            )
    return features


def _is_numerical(series: pd.Series) -> bool:
    """Test whether the series has only numerical values. There are two
    possibities, all the values are numbers are all the values are strings
    but are convertable to numbers. Therefore we first check wether is a number
    or a string and then we test whether the string is convertable to a number.
    """
    # Quick checks:
    if series.dtype in ["int64", "float"]:
        return True

    # Log rejected elements:
    if __debug__:
        return _all_elements_number(series)

    # Save time with using all (no logger tho):
    return all(
        _is_number(element)
        for element in series
    )


def _is_categorical(series: pd.Series) -> bool:
    """Test whether the series contains merely categories. Pandas has a built-
    in method for that. If the type is string this is also fine.
    """

    # Integers can both be interpreted as categories and as numerical.
    # We choose for them to be numerical. Therefore, if the feature is
    # numerical, it is not categorical
    if _is_numerical(series):  # Comment out for efficiency (UNSAFE)
        return False

    # Quick checks
    if series.dtype in ["category", "str"]:
        return True

    # Log rejected elements:
    if __debug__:
        return _all_elements_str(series)

    # Save time with using all (no logger tho):
    return all(
        isinstance(element, str)
        for element in series
    )


def _all_elements_str(series: pd.Series) -> bool:
    """Checks whether all element in de panda series are strings."""
    for element in series:
        if not isinstance(element, str):
            logger.info(
                f"{series.name} is not categorical because {element} with "
                f"with type {type(element)} is not a string"
            )
            return False
    return True  # All elements are strigns


def _all_elements_number(series: pd.Series) -> bool:
    for element in series:
        if not _is_number(element):
            logger.info(
                f"{series.name} is not numerical because {element} with "
                f"with type {type(element)} is not a string"
            )
            return False
    return True  # All ellements are numbers


def _is_number(value: Any) -> bool:  # noqa: ANN401
    try:
        float(value)
        return True
    except ValueError:
        return False


# The `else:` instances are superfluous but add semantic value
