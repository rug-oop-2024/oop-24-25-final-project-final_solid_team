#!/bin/env python

import numpy as np
from autoop.core.ml.model import get_model

MultipleLinearRegression = get_model("MultipleLinearRegression")
model = MultipleLinearRegression()

model.fit([[1]], [2])
print(model.predict([[1]]))