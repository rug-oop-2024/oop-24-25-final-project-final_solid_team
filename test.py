#!/bin/env python

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml, load_iris

from app.core.system import ArtifactRegistry, AutoMLSystem
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.storage import LocalStorage

artifact_storage = LocalStorage("./assets/dbo")
upper_level_storage = LocalStorage("./assets/objects")
artifact_database = Database(artifact_storage)
registry = ArtifactRegistry(
    storage=upper_level_storage,
    database=artifact_database
)

iris = load_iris()
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

dataset = Dataset.from_dataframe(
    name="test_dataset",
    data=df,
    asset_path="test_collection/test",
)

automl = AutoMLSystem.get_instance()
automl.registry.register(dataset)

datasets = automl.registry.list(type="dataset")
print(dataset)


