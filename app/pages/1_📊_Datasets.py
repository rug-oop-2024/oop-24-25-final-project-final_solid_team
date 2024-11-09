

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()

iris = load_iris()
iris_df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
iris_artifact = Dataset.from_dataframe(
    name="iris",
    data=iris_df,
    asset_path="datasets/iris"
)
automl.registry.register(iris_artifact)
datasets = automl.registry.list(type="dataset")


st.write(datasets[0].read())
