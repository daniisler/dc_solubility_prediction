# This script is there to test the best model obtained from the hyperparameter optimization from main.py. Using three different random seeds, the model is retrained three times and the final result is given as the average of the three runs, along with the standard deviation.

import os
import pandas as pd
import numpy as np
import pickle
import yaml
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from nn_model import SolubilityModel
import torch
from torch import nn, optim

from logger import logger
from data_prep import filter_temperature, calc_fingerprints, calc_rdkit_descriptors, gen_train_valid_test
from plot_config import *

import h5py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, RandomizedSearchCV, train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, RocCurveDisplay
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from scipy import stats
import importlib

# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
logger = logger.getChild('main_test')

# Input data file
input_type = 'Big'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_log.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)
cached_input_dir = os.path.join(PROJECT_ROOT, 'cached_input_data')
config_dir = os.path.join(PROJECT_ROOT, f'{input_type}_configs')
os.makedirs(cached_input_dir, exist_ok=True)

prediction_only = True

# Filter for solvents (list); A separate model is trained for each solvent in the list
if input_type == 'Aq':
    solvents = ['water']
elif input_type == 'Big':
    solvents = ['water', 'methanol', 'ethanol', 'toluene', 'chloroform', 'benzene', 'acetone'] #, 'toluene', 'chloroform', 'benzene', 'acetone'
# Filter for temperature in Kelvin; None for no filtering
T = 298
# Where to save the best model weights
model_save_folder = f'test_{input_type}_final'  # 'AqSolDB_filtered_fine'
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')
# Selected fingerprint for the model input
# Format fingerprint: (size, radius/(min,max_distance) respectively). If multiple fingerprints are provided, the concatenation of the fingerprints is used as input
selected_fp = {'m_fp': (128, 2), 'rd_fp': (128, (1,7)), 'ap_fp': (128, (1,30)), 'tt_fp': (128, 4)}  # Possible values: 'm_fp': (2048, 2), 'rd_fp': (2048, (1,7)), 'ap_fp': (2048, (1,30)), 'tt_fp': (2048, 4)
# Use additional rdkit descriptors as input
use_rdkit_descriptors = True
# List of rdkit descriptors to use; None or ['all'] for all descriptors
descriptors_list = ['all']
# Use additional descriptors from a DataFrame as input
use_df_descriptors = False
# List of DataFrame columns to use;
descriptors_df_list = []
# Missing value replacement for the rdkit descriptors
missing_rdkit_desc = 0.0
# Scale the input data
scale_transform = True
# Weight initialization method
weight_init = 'sTanh'  # 'target_mean', 'sTanh', 'Tanh', 'Tanshrink', 'default'
# Train/validation/test split
train_valid_test_split = [0.8, 0.1, 0.1]
# Random state for data splitting
random_states = [0, 1, 2]
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
mse_dict = {solvent: [] for solvent in solvents}
r2_dict = {solvent: [] for solvent in solvents}
mse_dict_dummy = {solvent: [] for solvent in solvents}
r2_dict_dummy = {solvent: [] for solvent in solvents}
for random_state in random_states:
    print(f'\nSTARTING EVALUATION FOR RANDOM STATE {random_state}\n')
    torch.manual_seed(random_state)
    # Define the hyperparameter grid; None if no training. In this case the model weights are loaded from the specified path. All parameters have to be provided in lists, even if only one value is tested
    logger.info(f'Evaluating the model performance for the solvent(s) {solvents}...')
    # Adapt the output directory for the current random state
    model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder, f'random_state_{random_state}')
    # Check if the output directory is empty
    os.makedirs(model_save_dir, exist_ok=True)
    if not len(os.listdir(model_save_dir)) == 0 and not prediction_only:
        overwrite = input(f'WARNING: The output directory {model_save_dir} is not empty, results might be overwritten. Do you want to continue? (y/N) ')
        if overwrite.lower() != 'y':
            raise SystemExit('User aborted the script...')
    # Train the model and evaluate the performance on the test dataset
    main_df = pd.read_csv(input_data_filepath)

    # Filter for room temperature
    if T:
        main_df = filter_temperature(main_df, T)
        if main_df.empty:
            raise ValueError(f'No data found for temperature {T} K. Exiting hyperparameter optimization.')

    # Create a new dataframe for each solvent
    df_list = [main_df[main_df['Solvent'] == solvent] for solvent in solvents]
    if any(df.empty for df in df_list):
        raise ValueError(f'No data found for {[solvent for solvent in solvents if df_list[solvents.index(solvent)].empty]} at T={T} K. Exiting hyperparameter optimization.')

    # Calculate the fingerprints or load them from cache (FIXME: Should remove it for production, but it speeds up the development process)
    for i, df in enumerate(df_list):
        cache_file = f'eval_cache/{input_type}_{solvents[i]}_{random_state}.h5'
        freshly_calc = False
        if not h5py.is_hdf5(cache_file):
            fingerprint_df_filename = f'{cached_input_dir}/{os.path.basename(input_data_filepath).split(".")[0]}_{selected_fp}_{solvents[i]}_{T}.csv'
            if os.path.exists(fingerprint_df_filename):
                logger.info(f'Loading fingerprints from {fingerprint_df_filename}')
                df_fp = pd.read_csv(fingerprint_df_filename)
                # Make a bitvector from the loaded bitstring
                for fp in selected_fp.keys():
                    df_fp[fp] = df_fp[fp].apply(lambda x: torch.tensor([int(c) for c in x], dtype=torch.float32))
            else:
                df_fp=calc_fingerprints(df_list[i], selected_fp=selected_fp)
                # Get the calculated fingerprints in a writeable format
                df_to_cache = df_fp.copy()
                df_to_cache.drop(columns=['mol', 'mol_solvent'], errors='ignore', inplace=True)
                for fp in selected_fp.keys():
                    df_to_cache[fp] = df_to_cache[fp].apply(lambda x: x.ToBitString())
                df_to_cache.to_csv(fingerprint_df_filename, index=False)

            # Calculate rdkit descriptors
            if use_rdkit_descriptors:
                df_fp, descriptor_cols = calc_rdkit_descriptors(df_fp, descriptors_list, missing_rdkit_desc)
            # Define the input and target data
            X = torch.tensor([])
            if len(selected_fp) > 0:
                X = torch.tensor(np.concatenate([df_fp[fp].values.tolist() for fp in selected_fp.keys()], axis=1), dtype=torch.float32)
            if use_rdkit_descriptors:
                descriptors_X = torch.tensor(df_fp[descriptor_cols].values.tolist(), dtype=torch.float32)
                X = torch.cat((X, descriptors_X), dim=1)
            y = torch.tensor(df_fp['Solubility'].values, dtype=torch.float32).reshape(-1, 1)

            # Split the data into train, validation and test set
            train_dataset, valid_dataset, test_dataset = gen_train_valid_test(X, y, model_save_dir=model_save_dir, solvent=solvents[i], split=train_valid_test_split, scale_transform=scale_transform, random_state=random_state)
            # Load the best hyperparameters config from the yaml file from wandb
            model_params_filename = os.path.join(config_dir, f'{solvents[i]}.yaml')
            with open(model_params_filename, 'r') as f:
                param_grid = yaml.safe_load(f)
            # Create an instance of our neural network
            import torch.nn.functional as F

            # Get the loss function from the string
            loss_fn_str = param_grid['loss_fn']['value']
            loss_fn = getattr(F, loss_fn_str.split('.')[-1])
            optimizer_str = param_grid['optimizer']['value']
            optimizer = getattr(optim, optimizer_str.split('.')[-1])
            activation_fn_str = param_grid['activation_fn']['value']
            activation_fn = getattr(nn, activation_fn_str.split('.')[-1])

            nn_model = SolubilityModel(
                batch_size=int(param_grid['batch_size']['value']),
                input_size=train_dataset.__X_size__(),
                n_neurons_hidden_layers=param_grid['n_neurons_hidden_layers']['value'],
                train_data=train_dataset,
                valid_data=valid_dataset,
                test_data=test_dataset,
                lr=float(param_grid['learning_rate']['value']),
                loss_function=loss_fn,
                optimizer=optimizer,
                activation_function=activation_fn,
                lr_factor=lr_factor,
                lr_patience=lr_patience,
                lr_threshold=lr_threshold,
                lr_min=lr_min,
                lr_mode=lr_mode,
                num_workers=num_workers
            )
            callbacks = []
            # Reset the early stopping callback
            if early_stopping:
                early_stop_callback = EarlyStopping(monitor="Validation loss", min_delta=ES_min_delta, patience=ES_patience, mode=ES_mode)
                callbacks.append(early_stop_callback)
            if restore_best_weights:
                checkpoint_callback = ModelCheckpoint(monitor="Validation loss", mode=ES_mode, save_top_k=1)
                callbacks.append(checkpoint_callback)
            # Define trainer
            trainer = Trainer(
                max_epochs=int(param_grid['max_epochs']['value']),
                callbacks=callbacks,
                enable_checkpointing=restore_best_weights,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",  # use GPU if available
            )
            if not prediction_only or not os.path.exists(os.path.join(model_save_dir, f'{solvents[i]}_model.pt')):
                # Initialize model weights and biases
                if weight_init != 'default':
                    nn_model.init_weights(weight_init)
                # Train the model
                trainer.fit(model=nn_model)
                # Load the best model
                if restore_best_weights:
                    nn_model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
                # Save the model for further evaluation
                model_filename = f'{model_save_dir}/{solvents[i]}_model.pt'
                torch.save(nn_model.state_dict(), model_filename)
                logger.info(f'Model saved to {model_filename}')
                # Save the model parameters
                model_params_filename = os.path.join(model_save_dir, f'{solvents[i]}_model_params.pkl')
                with open(model_params_filename, 'wb') as f:
                    pickle.dump(param_grid, f)
                logger.info(f'Model parameters saved to {model_params_filename}')
                # Save the scaler
                scaler_filename = os.path.join(model_save_dir, f'scaler_{solvents[i]}.pkl')
                with open(scaler_filename, 'wb') as f:
                    pickle.dump(train_dataset.scaler, f)
                # Validate the model
                val_loss = trainer.validate(model=nn_model)[0]['Validation loss']
                logger.info(f'Validation loss for {solvents[i]}: {val_loss}')
                # Test the model on the test dataset
                test_loss = trainer.test(model=nn_model)[0]['Test loss']
                logger.info(f'Test loss for {solvents[i]}: {test_loss}')
            else:
                nn_model.load_state_dict(torch.load(os.path.join(model_save_dir, f'{solvents[i]}_model.pt')))
            nn_model.eval()
            # Plot the true values vs. the predicted values
            predicted_values = nn_model(torch.Tensor(test_dataset.X))
            f = h5py.File(cache_file, 'w')
            true_values = test_dataset.y.detach().numpy()[:, 0]
            predicted_values = predicted_values.detach().numpy()
            mean_solubility = np.mean(train_dataset.y.detach().numpy())
            f.create_dataset('predicted_values', data=predicted_values)
            f.create_dataset('true_values', data=true_values)
            f.attrs['mean_solubility'] = mean_solubility
            f.close()
            freshly_calc = True

        # When the input is cached, load it from the cache
        if not freshly_calc:
            f = h5py.File(cache_file, 'r')
            predicted_values = f['predicted_values'][:]
            true_values = f['true_values'][:]
            mean_solubility = f.attrs['mean_solubility']
            f.close()
        mean_train = mean_solubility*np.ones_like(true_values)
        residuals = true_values-predicted_values
        assert residuals.shape == true_values.shape, f'Shape mismatch: {residuals.shape} vs. {true_values.shape}'

        # MSE and R2
        mse = np.mean(residuals**2)
        r2 = 1-np.sum(residuals**2)/np.sum((true_values-mean_train)**2)
        mse_dict[solvents[i]].append(mse)
        r2_dict[solvents[i]].append(r2)
        logger.info(f'Mean squared error for {solvents[i]}: {mse}')
        logger.info(f'R2 score for {solvents[i]}: {r2}')

        plt.plot(true_values, predicted_values, 'o')
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{input_type}DB_{solvents[i]}_true_vs_predicted_state_{random_state}.png'))
        plt.close()

        # Plot residuals vs. true values
        plt.plot(true_values, residuals, 'o')
        plt.plot([min(true_values), max(true_values)], [0, 0], 'r--')
        plt.xlabel('True values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{input_type}DB_{solvents[i]}_residuals_vs_true_state_{random_state}.png'))
        plt.close()

        # Plot a distribution of the residuals
        plt.hist(residuals, bins=10)
        plt.xlabel('Residuals')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{input_type}DB_{solvents[i]}_residuals_distribution_state_{random_state}.png'))
        plt.close()

        # Print the largest residuals
        largest_residuals = np.argsort(np.abs(residuals), axis=0)[-10:]
        # logger.info(f'Largest residuals for {solvents[i]}: {residuals[largest_residuals]}')

        # Compare to the model which would predict just the mean value
        residuals_mean = true_values-mean_train
        # MSE and R2 for the dummy model
        mse_mean = np.mean(residuals_mean**2)
        r2_mean = 1-np.sum(residuals_mean**2)/np.sum(residuals_mean**2)
        mse_dict_dummy[solvents[i]].append(mse_mean)
        r2_dict_dummy[solvents[i]].append(r2_mean)
        logger.info(f'Mean squared error for {solvents[i]}: {mse_mean} for the dummy model')
        logger.info(f'R2 score for {solvents[i]}: {r2_mean} for the dummy model')
        plt.plot(true_values, mean_train, 'o')
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
        plt.xlabel('True values')
        plt.ylabel('Dummy predicted values')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{input_type}DB_{solvents[i]}_true_vs_mean_predicted_state_{random_state}.png'))
        plt.close()
        plt.hist(residuals_mean, bins=10)
        plt.xlabel('Residuals')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'{input_type}DB_{solvents[i]}_residuals_distribution_mean_state_{random_state}.png'))
        plt.close()

        # TODO: Show the molecules with the largest residuals
        # for j in largest_residuals:
        #     logger.info(f'Molecule with the largest residual: {test_dataset.mols[j]}')
        #     logger.info(f'True value: {true_values[j]}, Predicted value: {predicted_values[j]}')

# Use the mse_dict and r2_dict to calculate the average and standard deviation
for solvent in solvents:
    mse_avg = np.mean(mse_dict[solvent])
    mse_std = np.std(mse_dict[solvent])
    r2_avg = np.mean(r2_dict[solvent])
    r2_std = np.std(r2_dict[solvent])
    mse_avg_dummy = np.mean(mse_dict_dummy[solvent])
    mse_std_dummy = np.std(mse_dict_dummy[solvent])
    r2_avg_dummy = np.mean(r2_dict_dummy[solvent])
    r2_std_dummy = np.std(r2_dict_dummy[solvent])
    # Print the results
    print(f'Average MSE for {solvent}: {mse_avg} +/- {mse_std}')
    print(f'Average R2 for {solvent}: {r2_avg} +/- {r2_std}')
    print(f'Average MSE for {solvent} (dummy model): {mse_avg_dummy} +/- {mse_std_dummy}')
    print(f'Average R2 for {solvent} (dummy model): {r2_avg_dummy} +/- {r2_std_dummy}')

