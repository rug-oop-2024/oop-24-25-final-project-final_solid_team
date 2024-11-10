#!/bin/env python

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml, load_iris

from app.core.system import ArtifactRegistry, AutoMLSystem
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.storage import LocalStorage

automl = AutoMLSystem.get_instance()
automl.registry.list()

# iris = load_iris()
# iris_df = pd.DataFrame(
#     data=iris.data,
#     columns=iris.feature_names
# )
# iris_artifact = Dataset.from_dataframe(
#     name="iris",
#     data=iris_df,
#     asset_path="datasets/iris"
# )
# automl.registry.register(iris_artifact)
# automl.registry.delete(iris_artifact.id)
