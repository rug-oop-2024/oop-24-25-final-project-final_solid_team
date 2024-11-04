from typing import Any, List
import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

logger = logging.getLogger(__name__)


def _is_categorical(candidate: Any):
    if isinstance(candidate, str):
        return True
    # TODO Also allow for one-hot encoding 


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
        if all(
            [isinstance(element, (float, int)) for element in df[column_name]]
        ):
            feature = Feature(
                type="numerical",
                name=column_name, 
                data=df[column_name]
            )
            features.append(feature)
        if all([_is_categorical(element) for element in df[column_name]]):
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
