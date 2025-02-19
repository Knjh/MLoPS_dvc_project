import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import yaml

# logging Module
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# init
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_ingestion.log')
fileHandler = logging.FileHandler(log_file_path)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(fileHandler)

# Load params
def load_params(params_path:str)->dict:
     try:
          with open(params_path, 'r') as file:
               params = yaml.safe_load(file)
          logger.debug('parms are extracted from %s:', params_path)
          return params
     except FileNotFoundError as e:
          logger.error('File not found: %s',e)
     except Exception as e:
          logger.error('Failed to fetch the parameters from:%s',e)


# ingestion
def load_data(data_path: str) -> pd.DataFrame:
     """" load data from csv file """
     try:
          df = pd.read_csv(data_path)
          logger.debug("data is succesfully loaded!")
          return df
     except pd.errors.ParserError as e:
          logger.debug("Failed to parse the file: %s", e)
          raise
     except Exception as e:
          logger.debug("Unexpected error occured: %s", e)
          raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
     """" save the train and test dataset """
     try:
          raw_data_path = os.path.join(data_path, 'raw')
          os.makedirs(raw_data_path, exist_ok=True)
          train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index = False)
          test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index = False)
          logger.debug("Data is succesfully saved in Folder: %s", raw_data_path)
     except Exception as e:
          logger.debug("Unexpected error occured while saving the data to: %s", e)
          raise
def main():
     try:
          # test_size = 0.2
          params = load_params('params.yaml')
          test_size = params['data_ingestion']['test_size']
          data_path = '/home/kanhaiya/MlopS/spam.csv'
          df = load_data(data_path)
          train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
          save_data(train_data, test_data, data_path='./data')
     except Exception as e:
          logger.debug("Failed to complete the data ingestion process: %s", e)
          print(f"error: {e}")

if __name__ == '__main__':
     main()