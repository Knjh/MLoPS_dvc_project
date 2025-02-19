import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle , yaml

# logging Module
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# init
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_training.log')
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

def model_train(x_train:np.ndarray, y_train:np.ndarray, params:dict) -> RandomForestClassifier:
     """ model training using random forest """
     try:
          if x_train.shape[0] != y_train.shape[0]:
               raise ValueError("input and output has different shape:")
          
          logger.debug("initializing random forest model:")
          clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

          logger.debug('model training started with %d sample:', x_train.shape[0])
          clf.fit(x_train, y_train)
          logger.debug('model training completed:')

          return clf
     except ValueError as e:
          logger.error('valuerror during traing: %s', e)
          raise
     except Exception as e:
          logger.error('error during training: %s', e)
          raise

def save_model(file_path: str, model) -> None:
     """ model saving """
     try:
          os.makedirs(os.path.dirname(file_path), exist_ok=True)

          with open(file_path, 'wb') as file:
               pickle.dump(model, file)
          logger.debug('model saved succesfully')
     except FileNotFoundError as e:
          logger.error('file not found: %s', file_path)
          raise
     except Exception as e:
          logger.error('error occured during model save: %s',e)
          raise

def main():
     try:
          params = load_params('params.yaml')
          params = params['model_building']

          train_data = load_data('/home/kanhaiya/MlopS/MLoPS_dvc_project/data/processed/train_tfidf.csv')
          x_train = train_data.iloc[:, :-1].values
          y_train = train_data.iloc[:,-1].values

          clf = model_train(x_train, y_train, params)

          model_save_path = 'models/model.pkl'
          save_model(model_save_path, clf)
     except Exception as e:
          logger.error('Failed to complete the training: %s', e)
          print(f'Error:{e}')

if __name__ == '__main__':
     main()