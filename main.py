# This is the main script for the project. It combines all the components

# TODO: Order us
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, LightningModule
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import itertools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json

from logger import logger
from data_prep import gen_train_valid_test, filter_temperature, calc_fingerprints
from nn_model import SolubilityModel
from hyperparam_optim import hyperparam_optimization
from dotenv import load_dotenv

# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
wandb_api_key = os.environ.get('WANDB_API_KEY', None)
logger = logger.getChild('main')

# Input parameters
input_data_filename = 'BigSolDB_filtered.csv'
# Filter for solvent; None for no filtering
solvent = 'methanol'
# Filter for temperature; None for no filtering
T = 293
# Where to write the results of the parameter optimization
output_paramoptim_file = 'hyperparam_optimization.json'
output_paramoptim_path = os.path.join(PROJECT_ROOT, 'logs', output_paramoptim_file)
# Wandb identifier
wandb_identifier = 'dc_solubility_prediction_test'
wandb_mode='online'
# Enable early stopping
early_stopping = True
ES_min_delta = 0.02
ES_patience = 5
ES_mode = 'min'
# Number of workers for data loading (recommended num_cpu_cores - 1)
num_workers = 7

# Define the hyperparameter grid
param_grid = {
    'batch_size': [16, 32, 64],
    'learning_rate': [0.0005, 0.001, 0.005],
    'n_neurons_hidden_layers': [[16], [32], [64], [128], [256], [32, 16], [64, 32]],
    'max_epochs': [50],
    'optimizer': [torch.optim.Adam], # torch.optim.SGD, torch.optim.Adagrad, torch.optim.Adamax, torch.optim.AdamW, torch.optim.RMSprop
    'loss_fn': [F.mse_loss], # F.mse_loss, F.smooth_l1_loss, F.l1_loss
    'activation_fn': [nn.ReLU], # nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU
}

# Check if the output file would be overwritten
if os.path.exists(output_paramoptim_path):
    raise FileExistsError(f'Output file {output_paramoptim_path} already exists. Please rename or delete it.')

# Load the (filtered) data from csv
# COLUMNS: SMILES,"T,K",Solubility,Solvent,SMILES_Solvent,Source
df = pd.read_csv(os.path.join(DATA_DIR, input_data_filename))

# Filter for room temperature
if T:
    df = filter_temperature(df, T)

# Filter for methanol solvent
if solvent:
    df = df[df['Solvent'] == solvent]

# Calculate the fingerprints
df = calc_fingerprints(df)

# Define the input and target data
X = torch.tensor(df['m_fp'].values.tolist(), dtype=torch.float32)
y = torch.tensor(df['Solubility'].values, dtype=torch.float32).reshape(-1, 1)

# Split the data into train, validation and test set
train_dataset, valid_dataset, test_dataset = gen_train_valid_test(X, y)

# Perform hyperparameter optimization
best_hyperparams, best_valid_score = hyperparam_optimization(param_grid, train_dataset, valid_dataset, test_dataset, wandb_mode=wandb_mode, wandb_identifier=wandb_identifier, early_stopping=early_stopping, ES_mode=ES_mode, ES_patience=ES_patience, ES_min_delta=ES_min_delta, wandb_api_key=wandb_api_key, num_workers=num_workers)

# Log the results to a json file
with open(output_paramoptim_path, 'w') as f:
    json.dump({'input_data_filename': input_data_filename, 'solvent': solvent, 'wandb_identifier': wandb_identifier, 'temperature': T, 'param_grid': param_grid, 'best_hyperparams': best_hyperparams, 'best_valid_score': best_valid_score}, f, indent=4)

logger.info(f'Hyperparameter optimization finished. Best hyperparameters: {best_hyperparams}, Best validation score: {best_valid_score}')

