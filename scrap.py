#!/bin/env python

from autoop.core.ml.model import regression
import inspect


for name, obj in inspect.getmembers(regression):
    print(type(obj))