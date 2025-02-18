import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ingestion
def load_data(data_path: str) -> pd.DataFrame:
     """" load data from csv file """
     try:
          df = pd.read_csv(data_path)
          df.fillna('',inplace=True)
          logger.debug("data is succesfully loaded!")
          return df
     except pd.errors.ParserError as e:
          logger.debug("Failed to parse the file: %s", e)
          raise
     except Exception as e:
          logger.debug("Unexpected error occured: %s", e)
          raise

def apply_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
     """ apply tfidf vectorization technique """
     try:
          X_train = train_data['preprocessed_text'].values
          X_test  = test_data['preprocessed_text'].values
          y_train = train_data['target'].values
          y_test  = test_data['target'].values

          tf = TfidfVectorizer()

          X_train_enc = tf.fit_transform(X_train)
          X_test_enc  = tf.transform(X_test)

          train_df = pd.DataFrame(X_train_enc.toarray())
          train_df['target'] = y_train
          test_df = pd.DataFrame(X_test_enc.toarray())
          test_df['target'] = y_test

          logger.debug('text vectorization has completed:')
          return train_df, test_df
     except Exception as e:
          logger.debug("Error during Tfidf Vectorization: %s", e)
          raise

def save_data(train_data: pd.DataFrame, data_path: str) -> None:
     """" save the train and test dataset """
     try:
          os.makedirs(os.path.dirname(data_path), exist_ok=True)
          train_data.to_csv(data_path, index = False)
          logger.debug("Data is succesfully saved in Folder: %s", data_path)
     except Exception as e:
          logger.debug("Unexpected error occured while saving the data to: %s", e)
          raise
def main():
     try:
          max_features = 50

          train_data = load_data('/home/kanhaiya/MlopS/MLoPS_dvc_project/src/data/interim/train_processed.csv')
          test_data  = load_data('/home/kanhaiya/MlopS/MLoPS_dvc_project/src/data/interim/test_processed.csv')

          train_df, test_df = apply_vectorization(train_data, test_data, max_features)

          save_data(train_df, os.path.join('./data', 'processed', 'train_tfidf.csv'))
          save_data(train_df, os.path.join('./data', 'processed', 'test_tfidf.csv'))

     except Exception as e:
          logger.log("Error occured in feature engineering module: %s", e)
          print(f'error: {e}')

if __name__ == '__main__':
     main()
