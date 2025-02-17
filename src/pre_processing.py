import pandas as pd
import os
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import logging
import string
punc = string.punctuation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# logging Module
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# init
logger = logging.getLogger('pre_processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'pre_processing.log')
fileHandler = logging.FileHandler(log_file_path)
fileHandler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(fileHandler)


def remove_punc(text):
    for char in punc:
        text = text.replace(char,'')
    return text

def stopwords_removal(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            ps.stem(word)
            new_text.append(word)
    return " ".join(new_text)


def preprocess_text(df, text_column='text', target_column='target'):
     """" preprocess the text """
     try:
          logger.debug('starting preprocessing for DataFrame')
          df['target'] = df['Category'].apply(lambda x: 0 if x == 'ham' else 1)
          logger.debug('Target columns encoded')

          # drop duplicates
          df.drop_duplicates(inplace = True)
          logger.debug("Duplicates removed")

          # cleaning
          df['preprocessed_text'] = df['Message'].apply(remove_punc)
          logger.debug("punctuation has removed")

          df['preprocessed_text'] = df['preprocessed_text'].apply(stopwords_removal)
          logger.debug("stopwards and stemming has done")

          return df

     except KeyError as e:
         logger.error('column not found: %s', e)
         raise
     except Exception as e:
         logger.error('Error during text normalization: %s', e)
         raise

def main(text_column = 'Message', target_column = 'Category'):
     """" main function to load the raw data, preprocess it and save the preprocessed data """
     try:
          # Fetch the data from data
          train_data = pd.read_csv('/home/kanhaiya/MlopS/src/data/train.csv')
          test_data = pd.read_csv('/home/kanhaiya/MlopS/src/data/train.csv')
          logger.debug('data loaded properly')

          # transform the data
          train_processed_data = preprocess_text(train_data, text_column, target_column)
          test_processed_data = preprocess_text(test_data, text_column, target_column)
          

          # store the data inside data/preprcessed
          data_path = os.path.join('./data', 'interim')
          os.makedirs(data_path, exist_ok=True)

          train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
          test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

          logger.debug('preprocessed data saved successfully: %s', data_path)

     except FileNotFoundError as e:
        logger.error('file not found: %s', e)
     except pd.errors.EmptyDataError as e:
        logger.error('data not found: %s', e)
     except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()