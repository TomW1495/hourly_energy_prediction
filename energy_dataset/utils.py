import kaggle

def download_dataset(dataset_name, destination):
    """"Downloads dataset from kaggle, requires API setup
        https://www.kaggle.com/docs/api"""
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_name, path=destination, unzip=True)

download_dataset("robikscube/hourly-energy-consumption", "data/")