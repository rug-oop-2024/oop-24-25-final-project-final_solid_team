import streamlit as st
from autoop.core.ml.metric import MeanSquaredError
import numpy as np

def main():
    file = st.file_uploader("")
    bytes = file.getvalue()
    print(bytes)
    # bytes = file.getvalue()

if __name__ == "__main__":
    main()

