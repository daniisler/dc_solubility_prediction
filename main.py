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
input_data_filename = 'BigSolDB_filtered.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)

# Filter for solvent; None for no filtering
solvent = 'methanol'
# Filter for temperature; None for no filtering
T = 293
# Where to write the results of the parameter optimization
output_paramoptim_file = 'hyperparam_optimization.json'
output_paramoptim_path = os.path.join(PROJECT_ROOT, 'logs', output_paramoptim_file)
os.makedirs(os.path.dirname(output_paramoptim_path), exist_ok=True)
# Where to save the best model weights
model_weights_file = 'best_weights_001.pth'
model_weigths_path = os.path.join(PROJECT_ROOT, 'saved_models', model_weights_file)
os.makedirs(os.path.dirname(model_weigths_path), exist_ok=True)
# Selected fingerprint for the model
# Format fingerprint: (size, radius/(min,max_distance) respectively). If multiple fingerprints are provided, the concatenation of the fingerprints is used as input
selected_fp = {'m_fp': (2048, 2), 'rd_fp': (1024, (1,7))} # 'm_fp': (2048, 2), 'rd_fp': (2048, (1,7)), 'ap_fp': (2048, (1,30)), 'tt_fp': 2048, 4)
scale_transform = True
# Train/validation/test split
train_valid_test_split = [0.8, 0.1, 0.1]
# Random state for data splitting
random_state = 0
# Wandb identifier
wandb_identifier = 'dc_solubility_prediction_test'
wandb_mode='disabled'
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
    'batch_size': [16],
    'learning_rate': [0.0005, 0.001],
    'n_neurons_hidden_layers': [[16]],
    'max_epochs': [50],
    'optimizer': [torch.optim.Adam], # torch.optim.SGD, torch.optim.Adagrad, torch.optim.Adamax, torch.optim.AdamW, torch.optim.RMSprop
    'loss_fn': [nn.functional.mse_loss], # nn.functional.mse_loss, nn.functional.smooth_l1_loss, nn.functional.l1_loss
    'activation_fn': [nn.ReLU], # nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU
}

# param_grid = None

if param_grid:
    # Loading all required modules takes some time -> only if needed
    from hyperparam_optim import hyperparam_optimization
    # Perform grid search on param_grid and save the results
    best_hyperparams = hyperparam_optimization(input_data_filepath=input_data_filepath, output_paramoptim_path=output_paramoptim_path, model_weigths_path=model_weigths_path, param_grid=param_grid, T=T, solvent=solvent, selected_fp=selected_fp, scale_transform=scale_transform, train_valid_test_split=train_valid_test_split, random_state=random_state, wandb_identifier=wandb_identifier, wandb_mode=wandb_mode, early_stopping=early_stopping, ES_mode=ES_mode, ES_patience=ES_patience, ES_min_delta=ES_min_delta, wandb_api_key=wandb_api_key, num_workers=num_workers)
