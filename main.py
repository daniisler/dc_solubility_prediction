# This is the main script for the project. It combines all the components and serves as an input file. Define the inputs here and execute this script to run the project.

import os
import pickle
from dotenv import load_dotenv

from logger import logger

# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
wandb_api_key = os.environ.get('WANDB_API_KEY', None)
logger = logger.getChild('main')

# Input parameters

# Set to True if only predictions should be made and no training is performed

# Set to True if only predictions should be made and no training is performed
prediction_only = False

# Input data file
input_type = 'Aq'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_descriptors_368.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)
cached_input_dir = os.path.join(PROJECT_ROOT, 'cached_input_data')
os.makedirs(cached_input_dir, exist_ok=True)

# Filter for solvents (list); A separate model is trained for each solvent in the list
solvents = ['water']  # ['methanol', 'ethanol', 'water', 'toluene', 'chloroform', 'benzene', 'acetone']#TODO
# Filter for temperature in Kelvin; None for no filtering
T = None
# Where to save the best model weights
model_save_folder = '_NN_rdkit_Big_water_298K/m_fp_A'  # 'AqSolDB_filtered_fine'
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')
# Selected fingerprint for the model input
# Format fingerprint: (size, radius/(min,max_distance) respectively). If multiple fingerprints are provided, the concatenation of the fingerprints is used as input
selected_fp = {'m_fp': (2048, 2)}  # Possible values: 'm_fp': (2048, 2), 'rd_fp': (2048, (1,7)), 'ap_fp': (2048, (1,30)), 'tt_fp': (2048, 4)
# Use additional rdkit descriptors as input
use_rdkit_descriptors = False
# List of rdkit descriptors to use; None or ['all'] for all descriptors
descriptors_list = ['MolLogP', 'LabuteASA', 'TPSA', 'MolWt']
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
# Wandb identifier
wandb_identifier = 'NN_rdkit_Big_water_298K'
wandb_mode = 'online' # 'disabled', 'offline'
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

# pylint: disable=wrong-import-position, wrong-import-order
import torch
from torch import nn, optim
torch.manual_seed(random_state)
# Define the hyperparameter grid; None if no training. In this case the model weights are loaded from the specified path. All parameters have to be provided in lists, even if only one value is tested
param_grid = {
    'batch_size': [16],  # 64, 256, 1024, 2048],
    'learning_rate': [1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 5e-8],
    'n_neurons_hidden_layers': [[60, 50, 40, 30, 20], [100, 80, 60, 40, 20], [200, 150, 100, 50, 20], [60, 50, 40], [40, 30, 20], [40, 30], [60, 30], [20, 40, 60, 100], [80, 50, 80, 50], [200], [10, 50, 10, 50, 10, 100]],
    'max_epochs': [5, 10, 30],
    'optimizer': [optim.RMSprop],  # optim.SGD, optim.Adagrad, optim.Adamax, optim.AdamW, optim.RMSprop, optim.Adam, optim.Adadelta
    'loss_fn': [nn.functional.mse_loss],  # nn.functional.mse_loss, nn.functional.smooth_l1_loss, nn.functional.l1_loss
    'activation_fn': [nn.Tanh],  # nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU
}

if param_grid and not prediction_only:
    logger.info(f'Performing hyperparameter optimization for the solvent(s) {solvents}...')
    logger.info(f'Using the following hyperparameter grid: {param_grid}')
    # Check if the output directory is empty
    os.makedirs(model_save_dir, exist_ok=True)
    if not len(os.listdir(model_save_dir)) == 0:
        overwrite = input(f'WARNING: The output directory {model_save_dir} is not empty, results might be overwritten. Do you want to continue? (y/N) ')
        if overwrite.lower() != 'y':
            raise SystemExit('User aborted the script...')
    # Loading all required modules takes some time -> only if needed
    from hyperparam_optim import hyperparam_optimization
    # Perform grid search on param_grid and save the results
    hyperparam_optimization(input_data_filepath=input_data_filepath, output_paramoptim_path=output_paramoptim_path, model_save_dir=model_save_dir, cached_input_dir=cached_input_dir, param_grid=param_grid, T=T, solvents=solvents, selected_fp=selected_fp, use_rdkit_descriptors=use_rdkit_descriptors, descriptors_list=descriptors_list, missing_rdkit_desc=missing_rdkit_desc, scale_transform=scale_transform, weight_init=weight_init, train_valid_test_split=train_valid_test_split, random_state=random_state, early_stopping=early_stopping, ES_mode=ES_mode, ES_patience=ES_patience, ES_min_delta=ES_min_delta, restore_best_weights=restore_best_weights, lr_factor=lr_factor, lr_patience=lr_patience, lr_threshold=lr_threshold, lr_min=lr_min, lr_mode=lr_mode, wandb_identifier=wandb_identifier, wandb_mode=wandb_mode, wandb_api_key=wandb_api_key, num_workers=num_workers)

# Check if the trained model weights exist
if not all(os.path.exists(os.path.join(model_save_dir, f'weights_{solvent}.pth')) for solvent in solvents):
    raise FileNotFoundError(f'Missing model weights in {model_save_dir} for solvent(s) {[solvent for solvent in solvents if not os.path.exists(os.path.join(model_save_dir, f"weights_{solvent}.pth"))]}!')
# Same for params
if not all(os.path.exists(os.path.join(model_save_dir, f'params_{solvent}.pkl')) for solvent in solvents):
    raise FileNotFoundError(f'Missing model parameters in {model_save_dir} for solvent(s) {[solvent for solvent in solvents if not os.path.exists(os.path.join(model_save_dir, f"params_{solvent}.pkl"))]}!')
# Same for scalers
if not all(os.path.exists(os.path.join(model_save_dir, f'scaler_{solvent}.pkl')) for solvent in solvents):
    raise FileNotFoundError(f'Missing scalers in {model_save_dir} for solvent(s) {[solvent for solvent in solvents if not os.path.exists(os.path.join(model_save_dir, f"scaler_{solvent}.pkl"))]}!')


# from predict import predict_solubility_from_smiles
# # Predict the solubility for the given SMILES
# smiles = 'c1cnc2[nH]ccc2c1'
# # Predict the solubility using a trained model, weights are loaded from the specified path and have to correspond to the best hyperparameters
# for solvent in solvents:
#     with open(os.path.join(model_save_dir, f'params_{solvent}.pkl'), 'rb') as f:
#         best_hyperparams = pickle.load(f)
#     solubility = predict_solubility_from_smiles(smiles, model_save_dir=model_save_dir, best_hyperparams=best_hyperparams, T=T, solvent=solvent, selected_fp=selected_fp, use_rdkit_descriptors=use_rdkit_descriptors, descriptors_list=descriptors_list, missing_rdkit_desc=missing_rdkit_desc, scale_transform=scale_transform)
#     logger.info(f'The predicted solubility for the molecule with SMILES {smiles} in {solvent} at T={T} K is {solubility}.')
