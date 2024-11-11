import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.datasets import fetch_openml, load_iris

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import (
    _is_categorical,
    _is_numerical,
    detect_feature_types,
)


class TestFeatures(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset_init_args = {
            "name": "test",
            "asset_path": "/tmp/tmp","data": b"test bytes string"
        }
        data = fetch_openml(name="adult", version=1, parser="auto")
        self.adult_df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.my_df = pd.DataFrame(
            columns=["ints",
            "categories",
            "floats",
            "string numbers",
            "wrong string numbers"],
            data=[
                [1, "a", 0.1, "1.12", "one"],
                [2, "b", 0.2, "555", "two"],
            ],  # TODO Make these numbers random
        )
        self.numerical_columns = [
            "age",
            "education-num",
            "capitalgain",
            "capitalloss",
            "hoursperweek",
        ]
        self.categorical_columns = [
            "education",
            "marital-status",
            "relationship",
            "race",
            "sex",
        ]

    def test_is_numerical(self):
        for column_name in self.numerical_columns:
            self.assertTrue(
                _is_numerical(self.adult_df[column_name])
            )
        for column_name in self.categorical_columns:
            self.assertFalse(
                _is_numerical(self.adult_df[column_name])
            )
        self.assertTrue(
            _is_numerical(self.my_df["ints"]))
        self.assertFalse(_is_numerical(self.my_df["categories"]))
        self.assertTrue(_is_numerical(self.my_df["string numbers"]))
        self.assertFalse(_is_numerical(self.my_df["wrong string numbers"]))

    def test_is_categorical(self):
        for column_name in self.numerical_columns:
            self.assertFalse(
                _is_categorical(self.adult_df[column_name]),
                msg=f"{self.adult_df[column_name].name} is deemed categorical",
            )
        for column_name in self.categorical_columns:
            self.assertTrue(
                _is_categorical(self.adult_df[column_name]),
                msg=(
                    f"{self.adult_df[column_name].name} is deemed not "
                    "categorical"
                ),
            )

    def test_unit_detect_features(self):
        data_set = Dataset(**self.dataset_init_args)
        df = pd.DataFrame(
            columns=["ints", "categories", "floats"],
            data=[
                [1, "a", 0.1],
                [2, "b", 0.2],
            ],
        )
        with patch("autoop.core.ml.dataset.Dataset.read") as read_mock:
            read_mock.return_value = df
            features = detect_feature_types(data_set)
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0].type, "numerical")
        self.assertEqual(features[1].type, "categorical")
        self.assertEqual(features[2].type, "numerical")

    def test_detect_features_continuous(self):
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")

    def test_detect_features_with_categories(self):
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 11)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "education",
            "marital-status",
            "relationship",
            "race",
            "sex",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in numerical_columns, features):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(lambda x: x.name in categorical_columns, features):
            self.assertEqual(detected_feature.type, "categorical")


# Notes: somehow the feature names of the `adult` data set do not correspond
# to given columns names with the difference being some dashes.

# Removed "workclass", "occupation" and "native country" from categorical types
#  because it contains NaNs
