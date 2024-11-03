#!/bin/env python

import pandas as pd
import numpy as np
from typing import List, Sized, Literal
import base64
from autoop.core.ml.artifact import Artifact

artifact = Artifact("database", "test_database")
artifact.save(b"hello world")
print(artifact.read())
