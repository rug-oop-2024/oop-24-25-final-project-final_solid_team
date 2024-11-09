import streamlit as st

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import MeanSquaredError, Metric
from autoop.core.ml.model import MultipleLinearRegression
from autoop.functional.feature import detect_feature_types
from app.core.system import AutoMLSystem


class PipelineHandler:
    """Convenient handler to facilate a page on a streamlit website to 
    create a pipeline."""
    def __init__(self) -> None:
        """Ask user for what kind of pipeline needs to be constructed."""
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
    
    def execute_pipeline(self) -> None:
        results = self._pipeline.execute()
        self._write_metric_results(results)

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
        return (features[0], features[1])
    
    def _choose_dataset(self) -> list[Dataset]:
        datasets = self._automl.registry.list(type="dataset")

        # Promote dataset artifact to Datasets
        # TODO Make it a for loop for better semantics
        [dataset.promote_to_subclass(Dataset)
         for dataset in datasets]
        
        return datasets[0]  # TODO Let the user choose