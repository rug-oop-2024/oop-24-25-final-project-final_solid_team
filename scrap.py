#!/bin/env python

import pandas as pd

from autoop.core.ml.artifact import Artifact

class Test:
    def __init__(self, *, x: int | None):
        self._x = x


x = Artifact()


x = pd.DataFrame()