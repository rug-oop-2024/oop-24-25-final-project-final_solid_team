from typing import List, Any

import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def _is_categorical(candidate: Any):
    if isinstance(candidate, str):
        return True


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
            feature = Feature(df[column_name], "numerical")
            features.append(feature)
        if all([_is_categorical(element) for element in df[column_name]]):
            feature = Feature(df[column_name], "categorical")
            features.append(feature)

        # No need to check for NaN types because only floats, ints and strings are added

    return features
