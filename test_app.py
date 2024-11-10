import streamlit as st
from autoop.core.ml.metric import MeanSquaredError
import numpy as np

if "counter" not in st.session_state:
    st.session_state["counter"] = 0

button = st.button("press me")

if button:
    st.session_state["counter"] += 1

st.write(st.session_state["counter"])

