import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


class DatasetHandler:
    """Handler to display datasets with streamlit app."""

    def __init__(self) -> None:
        """Handler to display datasets with streamlit app."""
        self._automl = AutoMLSystem.get_instance()

    def upload_csv_file(self) -> None:
        """Upload a file csv file and save it to the registry.
        """
        csv_file = st.file_uploader(
            label="Click here to upload a file",
        )

        if csv_file is not None:
            binary_data = csv_file.getvalue()

            dataset = Dataset(
                name=csv_file.name,
                asset_path=f"datasets/{csv_file.name}",
                data=binary_data,
            )

            self._automl.registry.register(dataset)

    def show_datasets(self) -> None:
        """Show the uploaded datasets.
        """
        datasets_artifacts = self._automl.registry.list(type="dataset")
        datasets = [artifact.promote_to_subclass(Dataset)
                    for artifact in datasets_artifacts]

        for dataset in datasets:
            checkbox = st.checkbox(f"Dataset {dataset.name}")
            if checkbox:
                st.write(dataset.read())
    
    def delete_datasets(self) -> None:
        datasets_artifacts = self._automl.registry.list(type="dataset")
        # TODO Make promotion of list a function
        datasets = [artifact.promote_to_subclass(Dataset)
                    for artifact in datasets_artifacts]
        

        if "deleting" not in st.session_state:
            st.session_state["deleting"] = False

        if st.button("Delete all datasets"):
            st.session_state["deleting"] = True

        if st.session_state["deleting"]:
            if st.button("Exit"):
                st.session_state["deleting"] = False
                st.rerun()

            # NOTE place holder functionality of streamlit does not work
            yes_no = st.selectbox(
                label="Are you sure you want to delete all datasets?",
                options=["choose an option", "yes", "no"],
            )
            if yes_no == "choose an option":
                st.write("you have not chosen yet")
            if yes_no == "yes":
                for dataset in datasets:
                    self._automl.registry.delete(dataset.id)
                    st.write(f"(Fake) Deleted {dataset.name}")
                    st.rerun()
            if yes_no == "no":
                st.write("Aborting deletions")


        # TODO Make an option to delete datasets!

# TODO Figure out why streamlit asked me to put a _ before self
