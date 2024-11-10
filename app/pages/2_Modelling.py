import streamlit as st
from autoop.core.ml.metric import MeanSquaredError
import numpy as np

# import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

from app.core.system import AutoMLSystem
from app.core.pipline_handler import PipelineHandler
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import MeanSquaredError, Metric

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a "
    "model on a dataset."
)

def _update_counter() -> None:
    if "counter" not in st.session_state:
        st.session_state["counter"] = 0
    button = st.button("Refresh?")
    if button:
        st.session_state["counter"] += 1

handler = PipelineHandler()
_update_counter()
handler.execute_pipeline(st.session_state["counter"])
