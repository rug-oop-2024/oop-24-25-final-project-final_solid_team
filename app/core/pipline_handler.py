from __future__ import annotations
import streamlit as st

from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from app.core.system import AutoMLSystem

def hash_handler(handler: PipelineHandler):
    pipeline = handler._pipeline
    


class PipelineHandler:
    """Convenient handler to facilate a page on a streamlit website to 
    create a pipeline."""
    def __init__(self) -> None:
        self._auto_ml_system = AutoMLSystem.get_instance()
        self._chosen_dataset = None

    def choose_dataset(self):
        all_artifacts = self._auto_ml_system.registry.list(type="dataset")

        chosen_artifact = st.selectbox(
            label="What data set would you like to use?",
            options=all_artifacts,
            format_func=lambda x: x.name,
            index=None
        )  # TODO Handle the case where there are no artifacts

        if chosen_artifact:
            self._chosen_dataset = chosen_artifact.promote_to_subclass(Dataset)

    def _select_features(self):
        # TODO Make sure output feature is not in the list of input features
        features = detect_feature_types(self._chosen_dataset)

        acceptable_input_features = filter(
            lambda x: x.type == "numerical",
            features
        )  # Only numerical input features are allowed
        chosen_features = st.multiselect(
            label="Select input features",
            options=acceptable_input_features,
            format_func=lambda x: x.name,
        )
        self._input_features = chosen_features

        self._output_feature = st.selectbox(
            label="Select output feature",
            options=features,
            format_func=lambda x: x.name,
        )  # Output feature can be anything.
        
        st.write(f"output: {self._output_feature.name}")

    def select_features(self):
        if "select features" not in st.session_state:
            st.session_state["select features"] = False
        
        if self._chosen_dataset:
            st.session_state["select features"] = True
        
        if st.session_state["select features"]:
            self._select_features()
        # features = detect_feature_types(self._chosen_dataset)
