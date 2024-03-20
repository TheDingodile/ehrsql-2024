import os
import json
import pandas as pd
from utils.data_io import read_json as read_data
from models.dummy_model import Model


# Directory paths for database, results and scoring program
DB_ID = 'mimic_iv'
BASE_DATA_DIR = 'sample_data'
RESULT_DIR = 'sample_result_submission/'
SCORE_PROGRAM_DIR = 'scoring_program/'

# File paths for the dataset and labels
TABLES_PATH = os.path.join('data', DB_ID, 'tables.json')               # JSON containing database schema
TRAIN_DATA_PATH = os.path.join(BASE_DATA_DIR, 'train', 'data.json')    # JSON file with natural language questions for training data
TRAIN_LABEL_PATH = os.path.join(BASE_DATA_DIR, 'train', 'label.json')  # JSON file with corresponding SQL queries for training data
VALID_DATA_PATH = os.path.join(BASE_DATA_DIR, 'valid', 'data.json')    # JSON file for validation data
DB_PATH = os.path.join('data', DB_ID, f'{DB_ID}.sqlite')               # Database path


train_data = read_data(TRAIN_DATA_PATH)
train_label = read_data(TRAIN_LABEL_PATH)
valid_data = read_data(VALID_DATA_PATH)


# Explore keys and data structure
# print(train_data.keys())
# print(train_data['version'])
# print(train_data['data'][0])

# Explore the label structure
print(train_label.keys())
print(train_label[list(train_label.keys())[0]])

myModel = Model()
data = valid_data["data"]

input_data = []
for sample in data:
    sample_dict = {}
    sample_dict['id'] = sample['id']
    sample_dict['input'] = sample['question']
    input_data.append(sample_dict)
    break
print(input_data)

label_y = myModel.generate(input_data)

print(label_y)
