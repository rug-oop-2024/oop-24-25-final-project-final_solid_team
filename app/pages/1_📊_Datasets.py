from app.core.dataset_handler import DatasetHandler

if __name__ == "__main__":
    handler = DatasetHandler()
    handler.upload_csv_file()
    handler.show_datasets()
