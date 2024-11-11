import os
import sys

app_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(app_dir)
sys.path.insert(0, project_dir)

import streamlit as st
from app.functional.streamlit import write_helper_text


def main() -> None:
    """Excecute main page. Inform the user how to use the app.
    """
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    write_helper_text(
        "Welcome to the Auto Machine Learning App."
        "Upload datasets in Datasets and craft your own pipeline in Modelling!"
    )


if __name__ == "__main__":
    main()
