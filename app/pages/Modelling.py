import streamlit as st

from app.core.pipline_handler import PipelineHandler


def main() -> None:
    """Set up moddeling page"""
    st.set_page_config(page_title="Modelling", page_icon="📈")

    def write_helper_text(text: str) -> None:
        """Inputs:
        text: str
        Writes text as helper text"""
        st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)

    st.write("# ⚙ Modelling")

    write_helper_text(
        "In this section, you can design a machine learning"
        " pipeline to train a model on a dataset."
    )

    # Would be way better in class. Time constraints...
    if "counter" not in st.session_state:
        st.session_state["counter"] = 0

    if "handler" not in st.session_state:
        st.session_state["handler"] = PipelineHandler()

    handler = st.session_state["handler"]
    # handler = PipelineHandler()
    handler.choose_dataset()
    handler.select_features()
    handler.ask_task_type()
    handler.choose_model()
    handler.choose_split()
    handler.choose_metric()
    handler.initialize_pipeline()
    handler.summary()
    handler.train()


if __name__ == "__main__":
    main()
