import streamlit as st

from app.core.pipline_handler import PipelineHandler
from app.functional.streamlit import write_helper_text

# Removed emoji and page number such that this module can be imported for
# testing.


def main() -> None:
    """Set up moddeling page"""
    st.set_page_config(page_title="Modelling", page_icon="📈")

    st.write("# ⚙ Modelling")

    write_helper_text(
        "In this section, you can design a machine learning"
        " pipeline to train a model on a dataset."
    )

    # Would be way better in class. Time constraints...
    if "counter" not in st.session_state:
        st.session_state["counter"] = 0

    if "results available" not in st.session_state:
        st.session_state["results available"] = False

    # Add pipeline to session state such pipeline handler will only be
    # initialized once
    if "handler" not in st.session_state:
        st.session_state["handler"] = PipelineHandler()

    handler = st.session_state["handler"]
    handler.choose_dataset()
    handler.select_features()
    handler.ask_task_type()
    handler.choose_model()
    handler.choose_split()
    handler.choose_metric()
    handler.initialize_pipeline()
    handler.summary()
    handler.train()
    handler.save()
    handler.handle_results()


if __name__ == "__main__":
    main()
