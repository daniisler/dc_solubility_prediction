import os
import json
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import itertools

from nn_model import SolubilityModel
from data_prep import gen_train_valid_test, filter_temperature, calc_fingerprints
from logger import logger

# Env
logger = logger.getChild('hyperparam_optimization')

# EXAMPLE PARAM_GRID
# param_grid = {
#     'batch_size': [16, 32, 64],
#     'learning_rate': [0.0005, 0.001, 0.005],
#     'n_neurons_hidden_layers': [[16], [32], [64], [128], [256], [32, 16], [64, 32]],
#     'max_epochs': [50]
# }


def hyperparam_optimization(input_data_filepath, output_paramoptim_path, model_save_dir, param_grid, T=None, solvents=None, selected_fp={'m_fp': (2048, 2)}, scale_transform=True, train_valid_test_split=[0.8, 0.1, 0.1], random_state=0, wandb_identifier='undef', wandb_mode='offline', early_stopping=True, ES_mode='min', ES_patience=5, ES_min_delta=0.05, wandb_api_key=None, num_workers=7):
    '''Perform hyperparameter optimization using grid search on the given hyperparameter dictionary.

    :param str input_data_filepath: path to the input data csv file
    :param str output_paramoptim_path: path to the output json file where the most important results are saved
    :param str model_weigths_path: path to the output file where the best model weights are saved
    :param dict param_grid: dictionary of hyperparameters to test, example see comment above
    :param float T: temperature used for filtering; None for no filtering
    :param str solvents: solvents used for filtering; None for no filtering
    :param dict of tuples selected_fp: selected fingerprint for the model, possible keys:
        - m_fp: Morgan fingerprint, tuple of (size, radius)
        - rd_fp: RDKit fingerprint, tuple of (size, (minPath, maxPath))
        - ap_fp: Atom pair fingerprint, tuple of (size, (min_distance, max_distance))
        - tt_fp: Topological torsion fingerprint, tuple of (size, torsionAtomCount)
        The selected fingerprints are calculated and concatenated to form the input data to the model
    :param bool scale_transform: whether to scale the input data
    :param list train_valid_test_split: list of train/validation/test split ratios, always 3 elements, sum=1
    :param int random_state: random state for data splitting for reproducibility
    :param str wandb_identifier: W&B project name
    :param str wandb_mode: W&B mode (online, offline, disabled, ...)
    :param bool early_stopping: enable early stopping
    :param str ES_mode: mode for early stopping
    :param int ES_patience: patience for early stopping
    :param float ES_min_delta: minimum delta for early stopping
    :param str wandb_api_key: W&B API key
    :param int num_workers: number of workers for data loading

    :return: best_hyperparams, saves the results to the output_paramoptim_path and the model weights to the model_weights_path

    '''

    # Check if the input file exists
    if not os.path.exists(input_data_filepath):
        raise FileNotFoundError(f'Input file {input_data_filepath} not found.')
    # Check if the output files would be overwritten
    if len(os.listdir(model_save_dir)) > 0:
        raise FileExistsError(f'The directory {model_save_dir} is not empty. Files could be overwritten! Please empty the directory or choose a different one.')

    # Load the (filtered) data from csv
    # COLUMNS: SMILES,"T,K",Solubility,Solvent,SMILES_Solvent,Source
    main_df = pd.read_csv(input_data_filepath)

    # Filter for room temperature
    if T:
        main_df = filter_temperature(main_df, T)
        if main_df.empty:
            raise ValueError(f'No data found for temperature {T} K. Exiting hyperparameter optimization.')

    # Create a new dataframe for each solvent
    if solvents:
        df_list = [main_df[main_df['Solvent'] == solvent] for solvent in solvents]
        if any([df.empty for df in df_list]):
            raise ValueError(f'No data found for {[solvent for solvent in solvents if df_list[solvents.index(solvent)].empty]} at T={T} K. Exiting hyperparameter optimization.')
    else:
        df_list = [main_df]
        solvents = ['all']

    # Calculate the fingerprints
    df_list_fp = [calc_fingerprints(df, selected_fp=selected_fp) for df in df_list]
    best_hyperparams_by_solvent = {}
    for i, df in enumerate(df_list_fp):
        # Define the input and target data
        X = torch.tensor(np.concatenate([df[fp].values.tolist() for fp in selected_fp], axis=1), dtype=torch.float32)
        y = torch.tensor(df['Solubility'].values, dtype=torch.float32).reshape(-1, 1)

        # Split the data into train, validation and test set
        train_dataset, valid_dataset, test_dataset = gen_train_valid_test(X, y, model_save_dir=model_save_dir, solvent=solvents[i], split=train_valid_test_split, scale_transform=scale_transform, random_state=random_state)

        # Perform hyperparameter optimization
        best_hyperparams, best_valid_score, best_model = grid_search_params(param_grid, train_dataset, valid_dataset, test_dataset, wandb_mode=wandb_mode, wandb_identifier=f'{wandb_identifier}_{solvents[i]}', early_stopping=early_stopping, ES_mode=ES_mode, ES_patience=ES_patience, ES_min_delta=ES_min_delta, wandb_api_key=wandb_api_key, num_workers=num_workers)

        # Convert the objects in the param grids (like nn.ReLu) to strings, so we can save them to a json file
        param_grid_str = param_grid.copy()
        best_hyperparams_by_solvent[solvents[i]] = best_hyperparams
        best_hyperparams_str = best_hyperparams.copy()
        for key, value in param_grid_str.items():
            param_grid_str[key] = [str(v) for v in value]
            best_hyperparams_str[key] = str(best_hyperparams[key])
        with open(f'{output_paramoptim_path.replace(".json", "")}_{solvents[i]}.json', 'w') as f:
            # Log the results to a json file
            json.dump({'input_data_filename': input_data_filepath, 'model_save_dir': model_save_dir, 'solvent': solvents[i], 'temperature': T, 'selected_fp': selected_fp, 'scale_transform': scale_transform, 'train_valid_test_split': train_valid_test_split, 'random_state': random_state, 'early_stopping': early_stopping, 'ES_mode': ES_mode, 'ES_patience': ES_patience, 'ES_min_delta': ES_min_delta, 'param_grid': param_grid_str, 'best_hyperparams': best_hyperparams_str, 'best_valid_score': best_valid_score, 'wandb_identifier': wandb_identifier}, f, indent=4)
            logger.info(f'Hyperparameter optimization finished. Best hyperparameters: {best_hyperparams}, Best validation score: {best_valid_score}, logs saved to {output_paramoptim_path}')
            # Save the best weights
            logger.info(f'Saving best weights to {model_save_dir}/weights_{solvents[i]}.pth')
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, f'weights_{solvents[i]}.pth'))

    return best_hyperparams_by_solvent


def grid_search_params(param_grid, train_data, valid_data, test_data, wandb_identifier, wandb_mode, early_stopping, ES_mode, ES_patience, ES_min_delta, wandb_api_key, num_workers):
    '''Perform hyperparameter optimization using grid search on the given hyperparameter dictionary.

    :param dict param_grid: dictionary of hyperparameters to test, example see comment above
    :param Dataset train_data: training dataset
    :param Dataset valid_data: validation dataset
    :param Dataset test_data: test dataset
    :param str wandb_identifier: W&B project name
    :param str wandb_mode: W&B mode (online, offline, disabled, ...)
    :param bool early_stopping: enable early stopping
    :param str ES_mode: mode for early stopping
    :param int ES_patience: patience for early stopping
    :param float ES_min_delta: minimum delta for early stopping
    :param str wandb_api_key: W&B API key (only required for online mode)
    :param int num_workers: number of workers for data loading

    :return: best hyperparameters (dict) and best validation score (float)

    '''
    best_score = np.inf
    best_hyperparams = {}
    best_model = None
    # Test all possible combinations of hyperparameters
    combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    for combination in combinations:
        logger.info(f"\n*** Run with hyperparameters: {combination} ***\n")
        # Start W&B
        wandb.finish()
        if not wandb_api_key and wandb_mode != 'offline':
            wandb_mode = 'offline'
            logger.warning('W&B API key not provided. Running in offline mode.')
        else:
            wandb.login(key=wandb_api_key, host='https://api.wandb.ai')
        wandb.init(project=wandb_identifier, config=combination, mode=wandb_mode)
        wandb_logger = WandbLogger()
        # Create an instance of our neural network
        nn_model = SolubilityModel(
            input_size=train_data.__X_size__(),
            n_neurons_hidden_layers=combination['n_neurons_hidden_layers'],
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            lr=combination['learning_rate'],
            batch_size=combination['batch_size'],
            optimizer=combination['optimizer'],
            loss_function=combination['loss_fn'],
            activation_function=combination['activation_fn'],
            num_workers=num_workers
        )
        # Reset the early stopping callback
        if early_stopping:
            early_stop_callback = EarlyStopping(monitor="Validation loss", min_delta=ES_min_delta, patience=ES_patience, mode=ES_mode)
        # Define trainer
        trainer = Trainer(
            max_epochs=combination['max_epochs'],
            logger=wandb_logger,
            callbacks=[early_stop_callback] if early_stopping else None,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",  # use GPU if available
        )
        # Train the model
        trainer.fit(model=nn_model)
        # Validate the model
        val_loss = trainer.validate(model=nn_model)[0]['Validation loss']
        # Update the best score and hyperparameters if current model is better
        if val_loss < best_score:
            best_score = val_loss
            best_hyperparams = combination
            best_hyperparams['n_epochs_trained'] = trainer.current_epoch
            best_model = nn_model
    wandb.finish()

    return best_hyperparams, best_score, best_model
