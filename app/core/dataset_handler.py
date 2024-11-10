import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


class DatasetHandler:
    """Handler to display datasets with streamlit app."""

    def __init__(_self) -> None:
        """Handler to display datasets with streamlit app."""
        _self._automl = AutoMLSystem.get_instance()

    def upload_csv_file(_self) -> None:
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

            _self._automl.registry.register(dataset)

    def show_datasets(_self) -> None:
        """Show the uploaded datasets.
        """
        datasets_artifacts = _self._automl.registry.list(type="dataset")
        datasets = [artifact.promote_to_subclass(Dataset)
                    for artifact in datasets_artifacts]

        for dataset in datasets:
            checkbox = st.checkbox(f"Dataset {dataset.name}")
            if checkbox:
                st.write(dataset.read())

        # TODO Make an option to delete datasets!

# TODO Figure out why streamlit asked me to put a _ before self
