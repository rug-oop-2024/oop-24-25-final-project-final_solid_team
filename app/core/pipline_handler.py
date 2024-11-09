from __future__ import annotations
import streamlit as st

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import MeanSquaredError, Metric
from autoop.core.ml.model import MultipleLinearRegression
from autoop.functional.feature import detect_feature_types
from app.core.system import AutoMLSystem

def hash_handler(handler: PipelineHandler):
    pipeline = handler._pipeline
    


class PipelineHandler:
    """Convenient handler to facilate a page on a streamlit website to 
    create a pipeline."""
    _number_of_calls = 0


    def __init__(self) -> None:
        """Ask user for what kind of pipeline needs to be constructed."""
        PipelineHandler._number_of_calls += 1
        self._automl = AutoMLSystem.get_instance()

        dataset = self._choose_dataset()
        
        features = detect_feature_types(dataset)
        input_features = self._choose_input_feature(features)
        target_feature = self._choose_target_feature(features)
        split = self._choose_split()

        self._pipeline = Pipeline(
            dataset=dataset,
            model=MultipleLinearRegression(),
            input_features=input_features,
            target_feature=target_feature,
            metrics=[MeanSquaredError()],
            split=split
        )

    # Add counter as argument such that everytime the counter is increased,
    # this function is executed again.
    @st.cache_resource
    def execute_pipeline(_self, counter_value) -> None:
        st.write("Executing the pipeline. Iteration "
                 f"{st.session_state['counter']}")
        results = _self._pipeline.execute()
        _self._write_metric_results(results)

    def _write_metric_results(self, results) -> None:
        evaluations: list[dict[Metric, float]] = results["metrics"]
        for metric, evaluation in evaluations:
            st.write(f"{metric.to_string()}: {evaluation}")
    
    def _write_predictions(self, results):
        check_box = st.checkbox("Show predictions")
        if check_box:
            st.write(results["predictions"])

    def _choose_split(self) -> float:
        return 0.8
    
    def _choose_target_feature(self, features: list[Feature]) -> Feature:
        return features[2]

    def _choose_input_feature(self, features: list[Feature]) -> list[Feature]:
        return st.multiselect(
            label="Which features do you want to use as input?",
            options=features,
            format_func=lambda x: x.name
        )
    
    def _choose_dataset(self) -> Dataset:
        artifacts = self._automl.registry.list(type="dataset")

        def get_name(dataset):
            return dataset.name

        chosen_artifact = st.selectbox(
            label="What data set would you like to use?",
            options=artifacts,
            format_func=get_name
        )
        
        return chosen_artifact.promote_to_subclass(Dataset)