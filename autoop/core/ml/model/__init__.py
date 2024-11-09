from __future__ import annotations

from typing import TYPE_CHECKING

from autoop.core.ml.model.regression import MultipleLinearRegression

if TYPE_CHECKING:
    from autoop.core.ml.model.model import Model

REGRESSION_MODELS = {
    "MultipleLinearRegression": MultipleLinearRegression
}  # add your models as str here

CLASSIFICATION_MODELS = {}  # add your models as str here

class ParameterException(Exception):
    pass


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        return REGRESSION_MODELS[model_name]
    if model_name in CLASSIFICATION_MODELS:
        return CLASSIFICATION_MODELS[model_name]
    raise ParameterException(
        f"{model_name} is not an available model of this package"
        )

# TODO Make it such that model are inaccesible except via get_model


