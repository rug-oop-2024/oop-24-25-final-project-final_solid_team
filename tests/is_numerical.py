#!/bin/env python
import base64
import logging
import sys
from typing import List, Literal, Sized
from numbers import Number

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from autoop.core.ml.feature import Feature

data = fetch_openml(name="adult", version=1, parser="auto")
df = pd.DataFrame(
    data.data,
    columns=data.feature_names,
)

def _is_numerical(series: pd.Series) -> bool:
    """Test whether the series has only numerical values. There are two
    possibities, all the values are numbers are all the values are strings
    but are convertable to numbers. Therefore we first check wether is a number
    or a string and then we test whether the string is convertable to a number.
    """
    return_value = True
    for element in series:
        if not isinstance(element, (Number, str)):  # TODO Fasten this process
            return_value = False
            logger.info(f"Series with name `{series.name}` is not numerical because "
                        f"`{element}` is not a Number nor a str")
            logger.debug(f"Failing series:\n{series}")
        if isinstance(element, str):
            if not element.isnumeric():
                logger.info(f"Series with name `{series.name}` is not numerical because "
                            f" `{element}` cannot be converted to a number")
                logger.debug(f"Failing series:\n{series}")
                return_value = False
                break
    return return_value

def test_total():
    data = fetch_openml(name="adult", version=1, parser="auto")
    df = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )
    for column_name in df:
        print(f"{df[column_name].name}, dtype: {df[column_name].dtype} -> {_is_numerical(df[column_name])}")

def test_education():
    data = fetch_openml(name="adult", version=1, parser="auto")
    df = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )
    print(f"education-num columns is numeric: {_is_numerical(df['education-num'])}")


def test_age():
    data = fetch_openml(name="adult", version=1, parser="auto")
    df = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )
    print(f"age columns is numeric: {_is_numerical(df['age'])}")

def test_not_convertable():
    df = pd.DataFrame(
        {"false": ["1", "2", "not a numberr"]}
    )
    print(f"non-converable series is numeric: {_is_numerical(df['false'])}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    # test_education()
    # test_age()
    # test_not_convertable()
    test_total()


