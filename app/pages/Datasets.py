from app.core.dataset_handler import DatasetHandler


def main() -> None:
    """Setup datasets page"""
    handler = DatasetHandler()
    handler.upload_csv_file()
    handler.show_datasets()
    handler.delete_datasets()


if __name__ == "__main__":
    main()
