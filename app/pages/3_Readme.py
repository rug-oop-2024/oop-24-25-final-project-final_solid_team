# from autoop.core.ml.artifact import Artifact
import streamlit as st

st.set_page_config(
    page_title="Readme",
    page_icon="👋",
)

st.markdown(open("README.md").read())
