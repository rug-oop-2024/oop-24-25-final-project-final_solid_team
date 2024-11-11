from __future__ import annotations

import logging
import sys

import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import (
    CLASSIFICATION_METRICS,
    METRICS,
    REGRESSION_METRICS
)
from autoop.core.ml.model import CLASSIFICATION_MODELS, REGRESSION_MODELS
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

logger = logging.getLogger()


class PipelineHandler:
    """Convenient handler to facilate a page on a streamlit website to
    create a pipeline."""
    def __init__(self) -> None:
        """Initialise the pipeline handeler"""
        self._auto_ml_system = AutoMLSystem.get_instance()
        self._pipeline = None
        self._chosen_dataset = None
        self._output_feature = None
        self._input_features = None
        self._task_type = None
        self._model = None
        self._model_name = None
        self._split = None
        self._metrics = None
        self._results = None
        self._pipeline_saved = False

        # Mechanism to allow for saving
        if "save pipeline" not in st.session_state:  # Assert exists
            st.session_state["save pipeline"] = False

    def save(self) -> None:
        """Save the pipeline."""
        if all((  # Make sure that pipeline has no uninitialized data
            self._chosen_dataset,
            self._output_feature,
            self._input_features,
            self._task_type,
            self._model,
            self._split,
            self._metrics,
            self._pipeline
        )):
            """ Saving the pipeline."""
            # Activate "save pipeline" mode
            if st.button(
                label="Save pipeline?"
            ):
                st.session_state["save pipeline"] = True

            # If activated, save the pipeline
            if st.session_state["save pipeline"]:
                name = st.text_input(
                    label="Enter pipeline name"
                )
                # Ask the user for name, will return None initially
                if name:
                    artifacts = self._pipeline.get_artifacts(name)
                    for artifact in artifacts:
                        self._auto_ml_system.registry.register(artifact)
                    self._pipeline_saved = True
                    # Deactivate "save pipeline mode"
                    st.session_state["save pipeline"] = False

        if self._pipeline_saved:
            st.write("Pipeline is saved")
    
    def handle_results(self) -> None:
        """Nicely write out the results."""
        # Obviously, get is way safer to acces session state items...
        if st.session_state.get("results available", None):
            metrics = self._results["metrics"]
            for metric, result in metrics:
                st.write(f"{metric.to_string()}: {result}")
        
            if st.checkbox("show prediction on test data"):
                st.write(self._results["test_predictions"])
            if st.checkbox("show prediction of training data"):
                st.write(self._results["train_predictions"])


    def train(self) -> None:
        """Train chosen model, using chosen variables by executing pipeline"""
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
            if st.button("Press to execute the pipeline"):
                self._results = self._pipeline.execute()
                st.session_state["results available"] = True

    def summary(self) -> None:
        """Summarize variables"""
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
            st.write(f"Model: {self._model_name}")
            st.write(f"Split: {self._split}")
            st.write(f"Metrics: {self._metrics}")

    def initialize_pipeline(self) -> None:
        """Initialize pipeling using chosen variables"""
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

    def choose_metric(self) -> None:
        """Produce options for choosing metric from available ones"""
        if self._split:
            options = (
                REGRESSION_METRICS if self._task_type == "regression"
                else CLASSIFICATION_METRICS
            )
            metrics_names = st.multiselect(
                label="Select a metric",
                options=options
            )
            if metrics_names:
                self._metrics = [METRICS[name]() for name in metrics_names]

    def choose_split(self) -> None:
        """Allow user to enter training/test split percentage"""
        if self._task_type:
            percentage = st.number_input(
                "Enter what percentage of the dataset will be used for "
                "training")
            self._split = percentage / 100

    def choose_model(self) -> None:
        """Produce options for choosing model from available ones"""
        if self._task_type:
            if self._task_type == "regression":
                model_name = st.selectbox(
                    label="Select a model.",
                    options=REGRESSION_MODELS,
                )
                if model_name:
                    self._model_name = model_name
                    self._model = REGRESSION_MODELS[model_name]
            if self._task_type == "classification":
                model_name = st.selectbox(
                    label="Select a model.",
                    options=CLASSIFICATION_MODELS
                )
                if model_name:
                    self._model_name = model_name
                    self._model = CLASSIFICATION_MODELS[model_name]

    def choose_dataset(self) -> None:
        """Produce options for chosing data set from available ones"""
        all_artifacts = self._auto_ml_system.registry.list(type="dataset")

        chosen_artifact = st.selectbox(
            label="Select dataset",
            options=all_artifacts,
            format_func=lambda x: x.name,
            index=None
        )  # TODO Handle the case where there are no artifacts

        if chosen_artifact:  # Can't promote NoneType to Dataset
            self._chosen_dataset = chosen_artifact.promote_to_subclass(Dataset)

    def select_features(self) -> None:
        """Ask the user to select features from a list of acceptable features.
        """
        if self._chosen_dataset:
            self._select_features()

    def ask_task_type(self) -> None:
        """Prompt the user with a box selection for which detection task
        he wants to use. If output feature is categorical, then only
        classification is possible. Otherwise both classification and
        regression is possible."""
        if self._output_feature and self._input_features:
            self._ask_task_type()

    def _select_features(self) -> None:
        """Produce options for choosing input and output features"""
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

    def _ask_task_type(self) -> None:
        """Produce option for choosing feature type"""
        if self._output_feature.type == "categorical":
            self._task_type = "classification"
            st.write("Task type will be classification because target "
                     "feature is categorical")
        if self._output_feature.type == "numerical":
            self._task_type = "regression"
            st.write("Task type will be classification because "
                     "target feature is numerical")

# TODO Declare all type of private member on top of the class
