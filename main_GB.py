import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
import optuna
from tqdm import tqdm
from logger import logger
from dotenv import load_dotenv

from data_prep import gen_train_valid_test, filter_temperature, calc_fingerprints
from gradient_boosting import objective, cv_model_optuna, gradient_boosting

# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
wandb_api_key = os.environ.get('WANDB_API_KEY', None)
logger = logger.getChild('main')

# Input data file
input_type = 'Aq'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_log.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)

# Where so save the best model weights
model_save_folder = 'test'
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')

# Select fingerprint for model
# TODO: implement other fingerprints, atm only m_fp works
selected_fp = {'m_fp': (2048, 2)}

# Select CV mode used (stratify for BigSolDB)
stratify = False

# Set parameters for CV
n_splits = 5
n_repeats = 1

# Random state for data splitting (only needed if stratify is False)
random_state = 0

# Choose which descriptors should be used as input for the model (give input as dictionary)
descriptors = None

# Choose max time for optimization in seconds
timeout = 30

# Choose name for study
study_name = 'testt'

gradient_boosting(
    input_data_filepath=input_data_filepath,
    output_paramoptim_path=output_paramoptim_path,
    model_save_dir=model_save_dir,
    selected_fp=selected_fp,
    descriptors=descriptors,
    stratify=stratify,
    n_splits=n_splits,
    n_repeats=n_repeats,
    timeout=timeout,
    random_state=random_state,
    study_name=study_name,
)
