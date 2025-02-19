import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle , json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
from dvclive import Live

# logging Module
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# init
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
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

def load_model(model_path: str):
     try:
          with open(model_path, 'rb') as file:
               model = pickle.load(file)
          logger.debug('model loaded successfully')
          return model
     
     except pd.errors.ParserError as e:
          logger.debug("Failed to parse the file: %s", e)
          raise
     except Exception as e:
          logger.debug("Unexpected error occured: %s", e)
          raise
def predict_eval(x_test: np.ndarray, y_test:np.ndarray, model)-> dict:
     try:
          y_pred = model.predict(x_test)
          y_pred_proba = model.predict_proba(x_test)[:, 1]

          acc = accuracy_score(y_test, y_pred)
          precision = precision_score(y_test, y_pred)
          recall = recall_score(y_test, y_pred)
          roc = roc_auc_score(y_test, y_pred_proba)

          metrics_dict = {
               'accuracy': acc,
               'precision': precision,
               'recall':recall,
               'roc_auc':roc
          }

          logger.debug("model evaluation completed:")
          return metrics_dict
     except Exception as e:
          logger.error('model evaluation failed: %s', e)
          raise

def save_metrics(file_path: str, metrics: dict) -> None:
     """ model saving """
     try:
          os.makedirs(os.path.dirname(file_path), exist_ok=True)

          with open(file_path, 'w') as file:
               json.dump(metrics, file, indent = 4)
          logger.debug('model saved succesfully')
     except FileNotFoundError as e:
          logger.error('file not found: %s', file_path)
          raise
     except Exception as e:
          logger.error('error occured during metrics save: %s',e)
          raise

def main():
     try:
          params = load_params('params.yaml')

          test_data = load_data('/home/kanhaiya/MlopS/MLoPS_dvc_project/data/preprocessed/test_tfidf.csv')
          x_test = test_data.iloc[:,:-1].values
          y_test = test_data.iloc[:,-1].values

          model_path = '/home/kanhaiya/MlopS/MLoPS_dvc_project/models/model.pkl'
          model = load_model(model_path)

          metrics = predict_eval(x_test, y_test, model)

          with Live(save_dvc_exp = True) as live:
               live.log_metric('accuracy:',metrics['accuracy'])
               live.log_metric('precission:',metrics['precision'])
               live.log_metric('recall:',metrics['recall'])

               live.log_params(params)

          save_metrics('reports/metrics.json', metrics)

     except Exception as e:
          logger.error('model evaluation failed: %s', e)
          print(f'error: {e}')

if __name__ == '__main__':
     main()







