from __future__ import annotations

import pickle
import sys
from typing import TYPE_CHECKING, List

import numpy as np

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features

# Moved Model import to hear because it is merely used for type checkign
# Also I removed it from model/__init__.py because user should not be able
# to acces abstract base class
if TYPE_CHECKING:
    from autoop.core.ml.model.model import Model

class Pipeline:
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        """Specify what data is used and how the model is trained and
        evuluated.

        Args:
            metrics (List[Metric]): List of Metric functions objects
            dataset (Dataset): Dataset
            model (Model): Model
            input_features (List[Feature]): input features
            target_feature (Feature): target feature
            split (float, optional): Train/test split. Defaults to 0.8.

        Raises:
            ValueError: Raises when data type and metric type do not match.
            Available types categorical/classification, continuous/regression
            for data/model respectively
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        self._check_same_type(target_feature, model)
        # TODO Bring back this check, error lays in spelling mistakes

    @staticmethod
    def _check_same_type(target_feature, model):
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification for categorical target "
                "feature\n"
                f"Feature {target_feature.name} with type "
                f" {target_feature.type} does not correspond to \n"
                f"Model {type(model)} with type {model.type}. Cause:\n"
                f"{target_feature.type} == categorical "
                f"and {model.type} != classification"
            )
        if (
            target_feature.type == "numerical"
            and model.type != "regression"
        ):
            print(target_feature.type, model.type, file=sys.stderr)
            raise ValueError(
                "Model type must be regression for continuous target "
                "feature\n"
                f"Feature {target_feature.name} with type "
                f" {target_feature.type} does not correspond to \n"
                f"Model {type(model)} with type {model.type}. Cause:\n"
                f"{target_feature.type} == numerical \n"
                f"or {model.type} != regression"
            )
        # TODO Bring back this check, error lays in spelling mistakes


    def __str__(self):
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self):
        return self._model  # UNSAFE, user can modify model.

    @property
    def artifacts(self) -> List[Artifact]:
        """Returns the artifacts of the
            - input feature numpy arrays
            - output feature numpy arrays
            - pipeline configurations
                - List[Features] of the input
                - List[Feature] of the output
                - float of the split
            - pipeline artifact
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact):
        self._artifacts[name] = artifact

    def preprocess_features(self):
        """
        Takes
            - self._input_features
            - self._target_feature
            - self._dataset
        and transforms it into:
            - self._input_vectors
            - self._output_vectors
        other than that it saves the
            - input Features
            - output Features
            -
        to self._artifact.
            and adds the encoding-artifacts of each feature to self._artifacts.
        During the transformation the features get encoded with either
        one-hot encoding or standard-scalar encoding.
        """
        # TODO Make proper docstring
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name for
        # consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self):
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)) :]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)) :
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self):
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate_on(self, X, Y):
        X = self._compact_vectors(X)
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric, result))
        return predictions

    def _evaluate(self):
        self._test_predictions = self._evaluate_on(self._test_X, self._test_y)
        self._train_predictions = self._evaluate_on(
            self._train_X, self._train_y
        )


    def execute(self) -> dict:
        """Executes the pipeline.

        Returns:
            dict: A dictionary with the following keys -> values
            - **"metrics"** -> **(list[tuple[Metric, float]])**: The value
                of the loss function of a specific metric.

            - **"test predictions"** -> **(np.ndarray)**: The predictions on
            the test dataset.
            - **"train predictions"** -> **(np.ndarray)**: The predictions on
            the train dataset.
        """
        self.preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "test_predictions": self._test_predictions,
            "train_predictions": self._train_predictions
        }

# Questions:
# - Why does __init__ get the data twice? Once via the Dataset, once via
#   input and output Features. Does Feature maybe not contain the data?
# What does an artifact in self._artifacts look like?
#   If the name of an artifact in self._artifacts is "OneHotEncoder" then
#   artifact["encoder"] gets has the data of the encoder.
# Why does property artifacts not return the datasets?
