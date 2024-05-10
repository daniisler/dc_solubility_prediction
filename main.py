# This is the main script for the project. It combines all the components and serves as an input file. Define the inputs here and execute this script to run the project.

import os

from logger import logger
from dotenv import load_dotenv

# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
wandb_api_key = os.environ.get('WANDB_API_KEY', None)
logger = logger.getChild('main')

# Input parameters
prediction_only = True

# Input data file
input_type = 'Big'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_log.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)

# Filter for solvents (list); None for no filtering, a model is trained for each solvent in the list
solvents = None #['methanol', 'ethanol']
# Filter for temperature in Kelvin; None for no filtering
T = None # 293
# Where to save the best model weights
model_save_folder = 'test_multi_model'
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')
os.makedirs(model_save_dir, exist_ok=True)
# Selected fingerprint for the model
# Format fingerprint: (size, radius/(min,max_distance) respectively). If multiple fingerprints are provided, the concatenation of the fingerprints is used as input
selected_fp = {'m_fp': (1024, 2)}  # Possible values: 'm_fp': (2048, 2), 'rd_fp': (2048, (1,7)), 'ap_fp': (2048, (1,30)), 'tt_fp': 2048, 4)
# Scale the input data
scale_transform = True
# Train/validation/test split
train_valid_test_split = [0.8, 0.1, 0.1]
# Random state for data splitting
random_state = 0
# Wandb identifier
wandb_identifier = 'test_multi_model'
wandb_mode = 'disabled'
# Enable early stopping
early_stopping = True
ES_min_delta = 0.02
ES_patience = 5
ES_mode = 'min'
# Number of workers for data loading (recommended num_cpu_cores - 1)
num_workers = 7

# Define the hyperparameter grid; None if no training. In this case the model weights are loaded from the specified path. All parameters have to be provided in lists, even if only one value is tested
import torch
import torch.nn as nn
param_grid = {
    'batch_size': [16], #, 32],
    'learning_rate': [5e-4], # [7e-4, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    'n_neurons_hidden_layers': [[64, 64, 32, 32]], #[[64, 64, 32, 16], [64, 48, 32], [64, 64, 64, 16], [64, 64, 32, 32], [64, 32, 16], [60, 50, 40, 30, 20], [60, 50, 40, 20, 10], [70, 60, 50, 40, 20], [70, 60, 50, 40, 30], [70, 60, 50, 40, 30, 20], [70, 60, 50, 40, 30, 20, 10]],
    'max_epochs': [1],
    'optimizer': [torch.optim.Adam],  # torch.optim.SGD, torch.optim.Adagrad, torch.optim.Adamax, torch.optim.AdamW, torch.optim.RMSprop
    'loss_fn': [nn.functional.mse_loss],  # nn.functional.mse_loss, nn.functional.smooth_l1_loss, nn.functional.l1_loss
    'activation_fn': [nn.ReLU],  # nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU
}

if param_grid and not prediction_only:
    # Loading all required modules takes some time -> only if needed
    from hyperparam_optim import hyperparam_optimization
    # Perform grid search on param_grid and save the results
    best_hyperparams = hyperparam_optimization(input_data_filepath=input_data_filepath, output_paramoptim_path=output_paramoptim_path, model_save_dir=model_save_dir, param_grid=param_grid, T=T, solvents=solvents, selected_fp=selected_fp, scale_transform=scale_transform, train_valid_test_split=train_valid_test_split, random_state=random_state, wandb_identifier=wandb_identifier, wandb_mode=wandb_mode, early_stopping=early_stopping, ES_mode=ES_mode, ES_patience=ES_patience, ES_min_delta=ES_min_delta, wandb_api_key=wandb_api_key, num_workers=num_workers)
else:
    best_hyperparams = None
    # TODO: Find a way to load the best hyperparameters from the hyperparameter output file
    # EXAMPLE:
    # best_hyperparams = {
    #     'batch_size': 16,
    #     'learning_rate': 5e-4,
    #     'n_neurons_hidden_layers': [64, 64, 32, 32],
    #     'max_epochs': 1,
    #     'optimizer': torch.optim.Adam,
    #     'loss_fn': nn.functional.mse_loss,
    #     'activation_fn': nn.ReLU,
    # }

# Check if the trained model weights exist
if not os.path.exists(os.path.join(model_save_dir, 'weights.pth')):
    raise FileNotFoundError(f'Weights not found at {model_save_dir}. Please train the model first.')

# Check if best_hyperparams are provided
if not best_hyperparams:
    raise ValueError('Please provide the best hyperparameters.')

from predict import predict_solubility_from_smiles
# Predict the solubility for the given SMILES
smiles = 'c1cnc2[nH]ccc2c1'
# Predict the solubility using a trained model, weights are loaded from the speficied path and have to
solubility = predict_solubility_from_smiles(smiles, model_save_dir=model_save_dir, best_hyperparams=best_hyperparams, T=T, solvents=solvents, selected_fp=selected_fp, scale_transform=scale_transform)
logger.info(f'The predicted solubility for the molecule with SMILES {smiles} in {solvents} at T={T} K is {solubility}.')
