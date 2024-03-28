import os
import json
import pandas as pd
from utils.data_io import read_json as read_data
from models.dummy_model import Model


# Directory paths for database, results and scoring program
DB_ID = 'mimic_iv'
BASE_DATA_DIR = 'data/mimic_iv'
RESULT_DIR = 'results/'
SCORE_PROGRAM_DIR = 'scoring_program/'

# File paths for the dataset and labels
TABLES_PATH = os.path.join('data', DB_ID, 'tables.json')               # JSON containing database schema
TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train', 'data.json')    # JSON file with natural language questions for training data
TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'train', 'label.json')  # JSON file with corresponding SQL queries for training data
VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, 'valid', 'data.json')    # JSON file for validation data
DB_PATH = os.path.join('data', DB_ID, f'{DB_ID}.sqlite')               # Database path
ANSWER_PATH = os.path.join(BASE_DATA_DIR, 'train', 'answer.json')      # JSON file with answers for training data

train_data = read_data(TRAIN_DATA_PATH)
train_label = read_data(TRAIN_LABEL_PATH)
answers = read_data(ANSWER_PATH)
valid_data = read_data(VALID_DATA_PATH)

myModel = Model()
data = train_data["data"]

print(data[:5])

# input_data = []
# for sample in data:
#     sample_dict = {}
#     sample_dict['id'] = sample['id']
#     sample_dict['input'] = sample['question']
#     input_data.append(sample_dict)
# print(input_data[:5])

label_y = myModel.generate(data)

# save predictios to results folder as json
with open(os.path.join(RESULT_DIR, 'ref', 'prediction.json', ), 'w') as f:
    json.dump(label_y, f)

