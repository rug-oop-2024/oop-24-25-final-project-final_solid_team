from app.core.dataset_handler import DatasetHandler


handler = DatasetHandler()
handler.upload_csv_file()
handler.show_datasets()
handler.delete_datasets()