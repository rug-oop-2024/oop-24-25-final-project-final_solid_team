import streamlit as st
from autoop.core.ml.metric import MeanSquaredError
import numpy as np


mse = MeanSquaredError()

string = mse.to_string()

st.write(string)



