#!/bin/env python

import streamlit as st
import numpy as np
from app.core.system import ArtifactRegistry
from autoop.core.database import Database
from autoop.core.storage import LocalStorage
from autoop.core.ml.dataset import Dataset

storage = LocalStorage()
database = Database(storage)
registry = ArtifactRegistry(
    storage=storage,
    database=database
)

dataset = Dataset(
    name="test dataset",
    asset_path="nov8/t21h59",
    data=b"hello world",
)

registry.register(dataset)


