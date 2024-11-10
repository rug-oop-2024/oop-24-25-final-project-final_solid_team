from __future__ import annotations
import streamlit as st

from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from app.core.system import AutoMLSystem


class PipelineHandler:
    """Convenient handler to facilate a page on a streamlit website to 
    create a pipeline."""
    def __init__(self) -> None:
        self._auto_ml_system = AutoMLSystem.get_instance()
        self._chosen_dataset = None
        self._output_feature = None
        self._input_features = None

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
    
    def select_features(self):
        """Ask the user to select features from a list of acceptable features.
        """
        if self._chosen_dataset:
            self._select_features()

    def ask_task_type(self):
        """Prompt the user with a box selection for which detection task
        he wants to use. If output feature is categorical, then only 
        classification is possible. Otherwise both classification and 
        regression is possible."""
        if self._output_feature and self._input_features:
            self._ask_task_type()

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
            index=None
        )  # Output feature can be anything.

    def _ask_task_type(self):
        st.write(self._output_feature.type)
        if self._output_feature.type == "categorical":
            self.task_type = "classification"
            st.write(
                "Since the output feature is categorical only classification "
                "is possible.")
        if self._output_feature.type == "numerical":
            selected_type = st.selectbox(
                label="Select the detection task",
                options=["classification", "numerical"],
            )
            if selected_type:
                self.task_type = selected_type




# TODO Declare all type of private member on top of the class