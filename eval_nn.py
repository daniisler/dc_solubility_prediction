
import os
import pickle
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from nn_model import SolubilityModel
from logger import logger

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
from torch import nn, optim
torch.manual_seed(random_state)


import json
import itertools

from data_prep import gen_train_valid_test, filter_temperature, calc_fingerprints, calc_rdkit_descriptors


# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT)
wandb_api_key = os.environ.get('WANDB_API_KEY', None)
logger = logger.getChild('main')


# Input data file
input_type = 'Aq'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_descriptors_368.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)
cached_input_dir = os.path.join(PROJECT_ROOT)
os.makedirs(cached_input_dir, exist_ok=True)

# Filter for solvents (list); A separate model is trained for each solvent in the list
solvents = ['water']  # ['methanol', 'ethanol', 'water', 'toluene', 'chloroform', 'benzene', 'acetone']
# Filter for temperature in Kelvin; None for no filtering
T = None

## TODO ADJUST:
identifier = 'NN_eval_1'
# Selected fingerprint for the model input
selected_fp = {'m_fp': (2048, 2)}  # Possible values: 'm_fp': (2048, 2), 'ap_fp': (2048, (1,30))
# Use additional rdkit descriptors as input
use_rdkit_descriptors = False
# List of rdkit descriptors to use; None or ['all'] for all descriptors
descriptors_list =  ['MolLogP', 'LabuteASA', 'TPSA', 'MolWt', 'FractionCSP3', 'BCUT2D_CHGLO','Kappa3','PEOE_VSA2', 'PEOE_VSA9']
# Use descriptors from data frame as input
use_df_descriptors = False
# Column names for descriptors to use
descriptors_df_list =  ['dipole', 'SASA']


# Define best hyperparameter
best_hyperparams = {
        'batch_size': [16],
        'learning_rate': [0.01],
        'n_neurons_hidden_layers': [200, 150, 100, 50, 20],
        'max_epochs': [250],
        'optimizer': [optim.RMSprop],
        'loss_fn': [nn.functional.smooth_l1_loss],
        'activation_fn': [nn.LeakyReLU],
    }



# Wandb identifier
wandb_identifier = identifier
wandb_mode = 'online'  #'disabled'#TODO
# Where to save the best model weights
model_save_folder = identifier
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')


# Missing value replacement for the rdkit descriptors
missing_rdkit_desc = 0.0
# Scale the input data
scale_transform = True
# Weight initialization method
weight_init = 'sTanh'  # 'target_mean', 'sTanh', 'Tanh', 'Tanshrink', 'default'
# Train/validation/test split
train_valid_test_split = [0.8, 0.1, 0.1]
# Random state for data splitting
random_state = 0

# Enable early stopping
early_stopping = True
ES_min_delta = 1e-4
ES_patience = 35
ES_mode = 'min'
restore_best_weights = True
# Learning rate scheduler (to deactivate set min_lr>=lr)
lr_factor = 0.3
lr_patience = 10
lr_threshold = 1e-3
lr_min = 1e-8
lr_mode = 'min'
# Number of workers for data loading (recommended less than num_cpu_cores - 1), 0 for no multiprocessing (likely multiprocessing issues if you use Windows and some libraries are missing); Specified in the .env file or as an environment variable
num_workers = int(os.environ.get('NUM_WORKERS', 0))


# Create a new dataframe for each solvent
df_list = [main_df[main_df['Solvent'] == solvent] for solvent in solvents]
if any(df.empty for df in df_list):
    raise ValueError(f'No data found for {[solvent for solvent in solvents if df_list[solvents.index(solvent)].empty]} at T={T} K. Exiting hyperparameter optimization.')

# Calculate the fingerprints or load them from cache (FIXME: Should remove it for production, but it speeds up the development process)
df_list_fp = []
for i, df in enumerate(df_list):
    fingerprint_df_filename = f'{cached_input_dir}/{os.path.basename(input_data_filepath).split(".")[0]}_{selected_fp}_{solvents[i]}_{T}.csv'
    if os.path.exists(fingerprint_df_filename):
        logger.info(f'Loading fingerprints from {fingerprint_df_filename}')
        df_list_fp.append(pd.read_csv(fingerprint_df_filename))
        # Make a bitvector from the loaded bitstring
        for fp in selected_fp.keys():
            df_list_fp[i][fp] = df_list_fp[i][fp].apply(lambda x: torch.tensor([int(c) for c in x], dtype=torch.float32))
    else:
        df_list_fp.append(calc_fingerprints(df_list[i], selected_fp=selected_fp))
        # Get the calculated fingerprints in a writeable format
        df_to_cache = df_list_fp[i].copy()
        df_to_cache.drop(columns=['mol', 'mol_solvent'], errors='ignore', inplace=True)
        for fp in selected_fp.keys():
            df_to_cache[fp] = df_to_cache[fp].apply(lambda x: x.ToBitString())
        df_to_cache.to_csv(fingerprint_df_filename, index=False)

# Calculate rdkit descriptors
if use_rdkit_descriptors:
    descriptor_cols_list = [None] * len(df_list_fp)
    for i, df in enumerate(df_list_fp):
        df_list_fp[i], descriptor_cols_list[i] = calc_rdkit_descriptors(df, descriptors_list, missing_rdkit_desc)

for i, df in enumerate(df_list_fp):
    X = torch.tensor([])
    if len(selected_fp) > 0:
        X = torch.tensor(np.concatenate([df[fp].values.tolist() for fp in selected_fp.keys()], axis=1), dtype=torch.float32)
    if use_rdkit_descriptors:
        descriptors_X = torch.tensor(df[descriptor_cols_list[i]].values.tolist(), dtype=torch.float32)
        X = torch.cat((X, descriptors_X), dim=1)
    if use_df_descriptors:
        if all(col in df.columns for col in descriptors_df_list):
            descriptors_df_X = torch.tensor(df[descriptors_df_list].values.tolist(), dtype=torch.float32)
            X = torch.cat((X, descriptors_df_X), dim=1)
            logger.info(f'Added DataFrame columns {descriptors_df_list} to input data X')
        else:
            missing_cols = [col for col in descriptors_df_list if col not in df.columns]
            logger.warning(f'Not all descriptors in descriptors_df_list are in DataFrame columns: None used.')

    y = torch.tensor(df['Solubility'].values, dtype=torch.float32).reshape(-1, 1)

    # Split the data into train, validation and test set
    train_dataset, valid_dataset, test_dataset = gen_train_valid_test(X, y, model_save_dir=model_save_dir, solvent=solvents[i], split=train_valid_test_split, scale_transform=scale_transform, random_state=random_state)



# Initialize Model
model = SolubilityModel(
    train_data=None,
    valid_data=None,
    test_data=None,
    input_size=sum(selected_fp[key][0] for key in selected_fp.keys()),
    n_neurons_hidden_layers=best_hyperparams['n_neurons_hidden_layers'],
    activation_function=best_hyperparams['activation_fn'],
    loss_function=best_hyperparams['loss_fn'],
    optimizer=best_hyperparams['optimizer'],
    lr=best_hyperparams['learning_rate'],
    batch_size=best_hyperparams['batch_size'],
    )
model.load_state_dict(torch.load(os.path.join(model_save_dir, f'weights_{solvent}.pth')))
model.eval()



Test_Set = np.concatenate(test_dataset, axis=1).reshape(1, -1)

# Predict the solubility
with torch.no_grad():
    Test_Set = torch.tensor(Test_Set, dtype=torch.float32)
    solubility = model(Test_Set).item()
