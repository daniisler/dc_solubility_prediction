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

# Choose name for study
study_name = 'test28'

# Select fingerprint for model
selected_fp = {'m_fp': (2048, 2)}  # Possible values: 'm_fp': (2048, 2), 'rd_fp': (2048, (1,7)), 'ap_fp': (2048,
# (1,30)), 'tt_fp': (2048, 4)

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

# Set parameters for lightgbm
lightgbm_params = None

# Settings for optuna.pruner
min_rescource = 'auto'
reduction_factor = 2
min_early_stopping_rate = 0
bootstrap_count = 0

# Settings for optuna study creation
direction = 'minimize'  # 'maximize' if higher scores are desired (used for different validation score calculations)
storage = 'sqlite:///db.sqlite3'  # use optuna-dashboard sqlite:///db.sqlite3 to look at runs

# Choose if some results should be printed in console
verbose = True

gradient_boosting(
    input_data_filepath=input_data_filepath,
    output_paramoptim_path=output_paramoptim_path,
    model_save_dir=model_save_dir,
    study_name=study_name,
    selected_fp=selected_fp,
    descriptors=descriptors,
    lightgbm_params=lightgbm_params,
    stratify=stratify,
    n_splits=n_splits,
    n_repeats=n_repeats,
    timeout=timeout,
    random_state=random_state,
    min_resource=min_rescource,
    reduction_factor=reduction_factor,
    min_early_stopping_rate=min_early_stopping_rate,
    bootstrap_count=bootstrap_count,
    direction=direction,
    storage=storage,
    verbose=verbose,
)
