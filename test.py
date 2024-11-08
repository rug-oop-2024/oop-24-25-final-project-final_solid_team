#!/bin/env python

import numpy as np
from autoop.core.ml.model.regression import MultipleLinearRegression


model = MultipleLinearRegression(
    hyper_params={
        "fit_intercept": False
    }
)

X = [
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
]

y = [
    [1, 2],
    [2, 4],
    [3, 6],
]

X_test = [
    [4, 4, 4],
    [5, 5, 5],
    [6, 6, 6],
]

model.fit(X, y)
pred = model.predict(X_test)


