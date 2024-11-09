

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

from app.core.system import AutoMLSystem
from app.core.handle_datasets import HandleDataset
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.dataset import Artifact

handler = HandleDataset()
handler.upload_csv_file()
handler.show_datasets()


