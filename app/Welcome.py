import os
import sys

app_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(app_dir)
sys.path.insert(0, project_dir)

import streamlit as st  # noqa: E402

from autoop.core.ml.artifact import Artifact  # noqa: E402

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Select a page above.")
st.markdown(open("README.md").read())


