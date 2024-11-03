#!/bin/env python

import base64
from typing import List, Literal, Sized

import numpy as np
import pandas as pd

from autoop.core.ml.artifact import Artifact

artifact = Artifact("database", "test_database")
artifact.save(b"hello world")
print(artifact.read())
