#!/bin/env python
import sys
import base64
from typing import List, Literal, Sized
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from autoop.core.ml.feature import Feature

from sklearn.datasets import fetch_openml
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

data = fetch_openml(name="adult", version=1, parser="auto")
df = pd.DataFrame(
    data.data,
    columns=data.feature_names,
)

help(is_categorical_dtype)

for column in df:
    is_categorical_dtype(column[df])
    print(f"Name {df[column].name}, Type: {df[column].dtype}"
        #   f"category? {is_categorical_dtype(column[df])}"
          f"numeric? {is_numeric_dtype(df[column])}")
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
#     test_log()



