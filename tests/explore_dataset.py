#!/bin/env python

from sklearn.datasets import fetch_openml
import pandas as pd


data = fetch_openml(name="adult", version=1, parser="auto")
df = pd.DataFrame(
    data.data,
    columns=data.feature_names,
)

print(df)

for column in df:
    print(f"{column}: "
          f"dtype={df[column].dtype} "
          f"type(element)={type(df[column][0])} ")