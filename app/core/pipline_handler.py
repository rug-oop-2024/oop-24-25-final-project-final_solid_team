from __future__ import annotations

import sys
import logging

import streamlit as st

from app.core.system import AutoMLSystem
from app.functional.streamlit import is_active
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.metric import REGRESSION_METRICS, CLASSIFICATION_METRICS
from autoop.core.ml.pipeline import Pipeline

logger = logging.getLogger()


class PipelineHandler:
    """Convenient handler to facilate a page on a streamlit website to 
    create a pipeline."""
    def __init__(self) -> None:
        self._auto_ml_system = AutoMLSystem.get_instance()
        self._pipeline = None
        self._chosen_dataset = None
        self._output_feature = None
        self._input_features = None
        self._task_type = None
        self._model = None
        self._split = None
        self._metrics = None

    def summary(self):
        if all((
            self._chosen_dataset,
            self._output_feature,
            self._input_features,
            self._task_type,
            self._model,
            self._split,
            self._metrics,
            self._pipeline
        )):
            st.write(f"Dataset: {self._chosen_dataset.name}")
            st.write("Input features: [")
            for feature in self._input_features:
                st.write(f"    {feature.name}")
            st.write("]")
            st.write(f"Output feature: {self._output_feature.name}")
            st.write(f"Task type: {self._task_type}")
            st.write(f"Model: {self._model}")
            st.write(f"Split: {self._split}")
            st.write(f"Metrics: {self._metrics}")

    def initialize_pipeline(self):
        # Check whether all variables are set
        print("Trying to initialized the pipeline.", file=sys.stderr)
        if all((
            self._chosen_dataset,
            self._output_feature,
            self._input_features,
            self._task_type,
            self._model,
            self._split,
            self._metrics
        )):
            self._pipeline = Pipeline(
                metrics=self._metrics,
                dataset=self._chosen_dataset,
                model=self._model(),  # Initialize model
                input_features=self._input_features,
                target_feature=self._output_feature,
                split=self._split,
            )
        
    def choose_metric(self):
        if self._split:
            options = (
                REGRESSION_METRICS if self._task_type == "regression"
                else CLASSIFICATION_METRICS
            )
            self._metrics = st.multiselect(
                label="Select a metric",
                options=options
            )

    def choose_split(self):
        if self._task_type:
            percentage = st.number_input(
                "Enter what percentage of the dataset will be used for "
                "training")
            self._split = percentage / 100
    
    def choose_model(self):
        if self._task_type:
            if self._task_type == "regression":
                model_name = st.selectbox(
                    label="Select a model.",
                    options=REGRESSION_MODELS,
                )
                st.write(model_name)
                if model_name:
                    self._model = REGRESSION_MODELS[model_name]
            if self._task_type == "classification":
                model_name = st.selectbox(
                    label="Select a model.",
                    options=CLASSIFICATION_MODELS
                )
                if model_name:
                    self._model = CLASSIFICATION_MODELS[model_name]
        if self._model:
            st.write(self._model)

    def choose_dataset(self):
        all_artifacts = self._auto_ml_system.registry.list(type="dataset")

        chosen_artifact = st.selectbox(
            label="Select dataset",
            options=all_artifacts,
            format_func=lambda x: x.name,
            index=None
        )  # TODO Handle the case where there are no artifacts

        if chosen_artifact:  # Can't promote NoneType to Dataset
            self._chosen_dataset = chosen_artifact.promote_to_subclass(Dataset)
            st.write(self._chosen_dataset)
    
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
        [st.write(str(feature)) for feature in self._input_features]

        self._output_feature = st.selectbox(
            label="Select output feature",
            options=features,
            format_func=lambda x: x.name,
            index=None
        )  # Output feature can be anything.
        if self._output_feature:
            st.write(self._output_feature.name)

    def _ask_task_type(self):
        if self._output_feature.type == "categorical":
            self._task_type = "classification"
            st.write("Task type will be classification because target "
                     "feature is categorical")
        if self._output_feature.type == "numerical":
            self._task_type = "regression"
            st.write("Task type will be classification because "
                     "target feature is numerical")

# TODO Declare all type of private member on top of the class