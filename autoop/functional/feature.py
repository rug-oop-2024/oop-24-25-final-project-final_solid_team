from typing import Any, List
import logging
from numbers import Number

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

logger = logging.getLogger(__name__)

def _is_numerical(series: pd.Series) -> bool:
    """Test whether the series has only numerical values. There are two
    possibities, all the values are numbers are all the values are strings
    but are convertable to numbers. Therefore we first check wether is a number
    or a string and then we test whether the string is convertable to a number.
    """
    return_value = True
    for element in series:
        if not isinstance(element, (Number, str)):  # TODO Fasten this process
            return_value = False
            logger.info(f"Series with name `{series.name}` is not numerical because "
                        f"`{element}` is not a Number nor a str")
            logger.debug(f"Failing series:\n{series}")
        if isinstance(element, str):
            if not element.isnumeric():
                logger.info(f"Series with name `{series.name}` is not numerical because "
                            f" `{element}` cannot be converted to a number")
                logger.debug(f"Failing series:\n{series}")
                return_value = False
                break
    return return_value

def _is_categorical(series: pd.Series) -> bool:
    """Test whether the series contains merely categories. Pandas has a built-
    in method for that. If the type is string this is also fine.
    NOTE:
        Sometimes integers are interpreted as categories by panda. Two
        solutions:
        1) First check for numerical within this method (SAFE)
        2) Implement this function such numerical is first checked (UNSAFE)
    """
    # Comment out for efficiency (UNSAFE)
    if _is_numerical(series):
        return False
    if series.dtype == "category":
        return True
    if series.dtype == "str":
        return True
    else:
        return False  # No category nor string type thus cannot be category

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    df = dataset.read()

    features = []
    for column_name in df:
        if _is_numerical(df[column_name]):
            feature = Feature(
                type="numerical",
                name=column_name, 
                data=df[column_name]
            )
            features.append(feature)
        if _is_categorical(df[column_name]):
            feature = Feature(
                type="categorical",
                name=column_name,
                data=df[column_name]
            )
            features.append(feature)
        else:
            logger.debug(
                f"Column {column_name} with type {df[column_name].dtype} is "
                "rejected"
            )
        # No need to check for NaN types because only floats, ints and strings are added

    return features
