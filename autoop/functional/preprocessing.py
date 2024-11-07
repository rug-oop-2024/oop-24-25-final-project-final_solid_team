from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def preprocess_features(
    features: List[Feature], dataset: Dataset
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Encode features by applying one-hot encoding to categorical features and
    standard-scalar encoding to numerical features.

    One-hot encoding converts all categories into binary arrays with all zeros
    except one.
    Standard-scalar encoding removes the mean and normalizes the standard
    deviation to one.

    Args:
        features (List[Feature]): List of features.
        dataset (Dataset): Dataset object.

    Returns:
        list[Tuple[str, np.ndarray, dict]]:
            A list of tuples where each tuple contains:
                - name (str): Name of the feature.
                - encoded_data (np.ndarray | OneHotArray): 
                    Data array of the feature with either scalar or one-hot
                    binary arrays.
                - artifact_dict (dict): Artifact dictionary with:
                    {
                        "type": "StandardScalar" | "OneHotEncoder",
                        "scalar" | "encoder" : dictionary with the settings
                                               (parameters) of the encoder.
                    }
    """
    results = []
    raw: pd.DataFrame = dataset.read()
    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
            ).toarray()
            aritfact = {
                "type": "OneHotEncoder",
                "encoder": encoder.get_params(),
            }
            results.append((feature.name, data, aritfact))
        if feature.type == "numerical":
            scaler = StandardScaler()
            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1)
            )
            artifact = {
                "type": "StandardScaler",
                "scaler": scaler.get_params(),
            }
            results.append((feature.name, data, artifact))
    # Sort for consistency
    results = list(sorted(results, key=lambda x: x[0]))
    return results


# Remark:
# Changed:
    # Returns:
    #     (List[str, Tuple[np.ndarray, dict]]): List of preprocessed features. 
    #     Each ndarray of shape (N, ...)
# into
    # Returns:
    #     (List[Tuple[str, np.ndarray, dict]]): List of preprocessed features. 
    #     Each ndarray of shape (N, ...)