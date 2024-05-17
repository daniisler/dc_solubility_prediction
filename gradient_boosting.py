import json

import os
import pandas as pd
import numpy as np
import optuna

from lightgbm import LGBMRegressor
from sklearn.model_selection import RepeatedKFold, GroupKFold
from collections import defaultdict
from tqdm import tqdm
from data_prep import calc_fingerprints
from sklearn.metrics import mean_squared_error
from logger import logger


# TODO: sort inputs
def gradient_boosting(
        input_data_filepath,
        output_paramoptim_path,
        model_save_dir=None,
        study_name: str = None,
        selected_fp=None,
        descriptors=None,
        lightgbm_params: dict = None,
        scale_transform: bool = True,  # TODO: Do I need this?
        stratify: bool = False,
        n_splits: int = 5,
        n_repeats: int = 1,
        timeout=3600,
        random_state: int = 0,
        min_resource='auto',
        reduction_factor: int = 2,
        min_early_stopping_rate: int = 0,
        bootstrap_count: int = 0,
        direction: str = 'minimize',
        storage: str = 'sqlite:///db.sqlite3',
        verbose: bool = True
):
    """
    Build gradient boosting model and optimze its hyperparameters using optuna.
    :param str input_data_filepath: path to the input data csv file
    :param str output_paramoptim_path: path to the output json file where the most important results are saved
    :param str model_save_dir: path to the output file where the best model weights are saved
    :param str study_name: name of study
    :param dict of tuples selected_fp: selected fingerprint for the model, possible keys:
        - m_fp: Morgan fingerprint, tuple of (size, radius)
        - rd_fp: RDKit fingerprint, tuple of (size, (minPath, maxPath))
        - ap_fp: Atom pair fingerprint, tuple of (size, (min_distance, max_distance))
        - tt_fp: Topological torsion fingerprint, tuple of (size, torsionAtomCount)
    :param dict descriptors: selected descriptors for the model
    :param dict lightgbm_params: parameters used for by lightgbm
    :param bool stratify: decides if stratified CV odr normal CV is used (Smiles is stratified variable)
    :param int n_splits: number of splits used for CV
    :param int n_repeats: number of times CV is repeated
    :param timeout: stop study after given number of seconds
    :param int random_state: random state for data splitting for reproducibility
    :param min_resource: Parameter for specifying the minimum resource allocated to a trial. This parameter defaults
    to ‘auto’ where the value is determined based on a heuristic that looks at the number of required steps for the first trial to complete.
    :param int reduction_factor: Parameter for specifying reduction factor of promotable trials
    :param int min_early_stopping_rate: Parameter for specifying the minimum early-stopping rate
    :param int bootstrap_count: Parameter specifying the number of trials required in a rung before any trial can be promoted.
    :param str direction: Direction of optimization, 'minimize' or 'maximize'
    :param str storage: URL for database
    :param bool verbose: print mse and other info in console or not
    """

    if selected_fp is None:
        selected_fp = {'m_fp': (2048, 2)}

    if lightgbm_params is None:
        lightgbm_params = {
            'num_leaves': (70, 150),  # Number of leaves in a tree
            # 'learning_rate': (0.005, 0.1),  # Learning rate
            'n_estimators': (700, 1500),  # Number of boosting rounds
            'max_depth': (10, 30),  # Maximum tree depth
            # 'min_child_samples': (5, 40),  # Minimum number of data in a leaf
            'subsample': (0.5, 1),  # Subsample ratio of the training data
            'colsample_bytree': (0.5, 1),  # Subsample ratio of columns when constructing each tree
            # 'extra_trees': [True, False],
        }

    # Check if the input file exists
    if not os.path.exists(input_data_filepath):
        raise FileNotFoundError(f'Input file {input_data_filepath} not found.')

    # Read input data (and filter lines with '-' as SMILES_Solvent)
    df = pd.read_csv(input_data_filepath)
    df = df[df['SMILES_Solvent'] != '-']

    # Calculate molecule object and fingerprints for solutes and solvents and rename column
    df = calc_fingerprints(df=df, selected_fp=selected_fp, solvent_fp=True)
    df.rename(columns={list(selected_fp.keys())[0]: f'{list(selected_fp.keys())[0]}_mol'}, inplace=True)

    # Calculate descriptors
    desc_cols = []
    if descriptors:
        logger.info(f'Calculating descriptors:{list(descriptors.keys())[0]}')
        for col in ['mol', 'mol_solvent']:
            for desc_name, desc_func in descriptors.items():
                df[f"{col}_{desc_name}"] = df[col].apply(desc_func)
                desc_cols.append(f"{col}_{desc_name}")

        logger.info(f'Descriptors used as features: {desc_cols}')

    # Make a new column in df for every element in fingerprint list (easier format to handle)
    mol_fingerprints = np.stack(df[f'{list(selected_fp.keys())[0]}_mol'].values)
    df_mol_fingerprints = pd.DataFrame(mol_fingerprints, columns=[f'mol_fp_{i}' for i in range(mol_fingerprints.shape[1])])

    solvent_fingerprints = np.stack(df[f'{list(selected_fp.keys())[0]}_solvent'].values)
    df_solvent_fingerprints = pd.DataFrame(mol_fingerprints, columns=[f'solvent_fp_{i}' for i in range(solvent_fingerprints.shape[1])])

    # Reset indices of df so pd.concat does not produce any NaN
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df_mol_fingerprints, df_solvent_fingerprints], axis=1)

    # Create list of feature and target columns
    fp_cols = list(df_mol_fingerprints.columns) + list(df_solvent_fingerprints.columns)
    target_col = 'Solubility'
    feature_cols = fp_cols + desc_cols

    # Add temperature as feature, if stratified CV is used
    if stratify:
        feature_cols = feature_cols + ['T,K']

    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=min_resource, reduction_factor=reduction_factor,
                                                    min_early_stopping_rate=min_early_stopping_rate,
                                                    bootstrap_count=bootstrap_count)

    # Create a study object and optimize
    study = optuna.create_study(direction=direction, pruner=pruner, storage=storage,
                                study_name=study_name)
    study.optimize(
        lambda trial: objective(trial, df=df, feature_cols=feature_cols, target_col=target_col, n_splits=n_splits, n_repeats=n_repeats,
                                random_state=random_state, lightgbm_params=lightgbm_params, stratify=stratify, verbose=verbose),
        timeout=timeout)

    # Make descriptors an empty dict if its None, so json file can be written normally
    if descriptors is None:
        descriptors = {}
    (os.makedirs(model_save_dir, exist_ok=True))
    with (open(output_paramoptim_path, 'w', encoding='utf-8') as f):
        # Log the results to a json file
        json.dump({'input_data_filename': input_data_filepath, 'output_paramoptim_path': output_paramoptim_path,
                   'model_save_dir': model_save_dir, 'study_name': study_name, 'selected_fp': selected_fp,
                   'descriptors': list(descriptors.keys()), 'lightgbm_params': lightgbm_params, 'scale_transform':
                       scale_transform, 'stratify': stratify, 'n_splits': n_splits, 'n_repeats': n_repeats,
                   'timeout': timeout, 'random_state': random_state, 'min_resource': min_resource,
                   'reduction_factor': reduction_factor, 'min_early_stopping_rate': min_early_stopping_rate,
                   'bootstrap_count': bootstrap_count, 'direction': direction, 'storage': storage,
                   'best_hyperparamers': study.best_params, 'best_value': study.best_value,
                   'best_value_sd': study.best_trial.user_attrs['mse_sd']}, f,
                  indent=4)

    logger.info(f'Hyperparameter optimization finished. Best hyperparameters: {study.best_params}, best mse: {study.best_value}')

    return study.best_params


def cv_model_optuna(
        trial,
        model,
        df,
        feature_cols: list = None,
        target_col: str = 'Solubility',
        metric_fs=None,
        n_splits: int = 6,
        n_repeats: int = 4,
        random_state: int = 0,
        verbose: bool = True,
        stratify: bool = False,
):
    """
    Perform RepeatedKFold or Stratified cross validation of a model  and calculate specified metrics on the oof
    predictions.
    :param trial: object of the current trial
    :param model: type of model used
    :param df: dataframe of input data
    :param list feature_cols: list of the column names of the features
    :param str target_col: column name of target
    :param metric_fs:
    :param int n_splits: number of splits used for CV
    :param int n_repeats: number of times CV is repeated
    :param int random_state: random state for data splitting for reproducibility
    :param bool verbose: print mse and other info in console or not
    :param bool stratify: decides if stratified CV odr normal CV is used (Smiles is stratified variable)
    In addition to cv_model this function reports intermediate values to a gradient_boosting pruner and prunes the trial if needed.
    """
    metric_fs = metric_fs if metric_fs is not None else {'mse': mean_squared_error}

    # Use GroupKFold or RepeatedKFold for CV
    if stratify:
        # Use SMILES column as stratify variable
        groups = df['SMILES']

        # Convert pandas dfs to numpy.ndarrays for better performance, consider using .to_numpy instead
        X = df[feature_cols].values
        y = df[target_col].values

        gkf = GroupKFold(n_splits=n_splits)
        folds = gkf.split(X, y, groups=groups)

    else:
        # Convert pandas dfs to numpy.ndarrays for better performance, consider using .to_numpy instead
        X = df[feature_cols].values
        y = df[target_col].values

        skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        folds = skf.split(X, y)

    metrics_list = defaultdict(list)
    # loop over different folds (tqdm adds progress bar to loop)
    for idx, (train_index, val_index) in tqdm(enumerate(folds)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train.reshape(-1, 1))
        y_pred = model.predict(X_val)

        for metric_name, metric_f in metric_fs.items():
            metrics_list[metric_name].append(metric_f(y_val.ravel(), y_pred.ravel()))

        if verbose:
            print(f"mean mse: {np.mean(metrics_list['mse'])} and {trial.params}")

        trial.report(np.mean(metrics_list['mse']), idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    metrics = {k: (np.mean(v), np.std(v)) for k, v in metrics_list.items()}

    logger.info(f"Trial finished with mean mse: {np.mean(metrics_list['mse'])}, mse_std: {np.std(metrics_list['mse'])} and parameters:"
                f" {trial.params}")

    if verbose:
        print(f"Trial finished with mean mse: {np.mean(metrics_list['mse'])}, mse_std: {np.std(metrics_list['mse'])} and parameters:"
              f" {trial.params}")
    return metrics


def objective(trial, df, feature_cols, target_col, n_splits: int = 5, n_repeats: int = 4, lightgbm_params=None,
              random_state: int = 0,
              verbose=True, stratify: bool = False
              ):
    """
    :param trial: object of the current trial
    :param df: dataframe of input data
    :param list feature_cols: list of the column names of the features
    :param str target_col: column name of target
    :param int n_splits: number of splits used for CV
    :param int n_repeats: number of times CV is repeated
    :param dict lightgbm_params: parameters used for lightgbm
    :param int random_state: random state for data splitting for reproducibility
    :param bool verbose: print mse and other info in console or not
    :param bool stratify: decides if stratified CV odr normal CV is used (Smiles is stratified variable)
    In addition to cv_model this function reports intermediate values to a gradient_boosting pruner and prunes the trial if needed.
    """

    # Sample hyperparameters
    params = {
        'num_leaves': trial.suggest_int('num_leaves', *lightgbm_params['num_leaves']),
        # 'learning_rate': trial.suggest_loguniform('learning_rate', *lightgbm_params['learning_rate']),
        'n_estimators': trial.suggest_int('n_estimators', *lightgbm_params['n_estimators']),
        'max_depth': trial.suggest_int('max_depth', *lightgbm_params['max_depth']),
        # 'min_child_samples': trial.suggest_int('min_child_samples', *lightgbm_params['min_child_samples']),
        'subsample': trial.suggest_float('subsample', *lightgbm_params['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *lightgbm_params['colsample_bytree']),
        # 'extra_trees': trial.suggest_categorical('extra_trees', lightgbm_params['extra_trees'])
    }

    # Create estimator
    model = LGBMRegressor(**params, verbose=-1)  # verbose=int(verbose) behaves differently

    # Evaluate model
    metrics = cv_model_optuna(trial, model, df=df, feature_cols=feature_cols, target_col=target_col,
                              n_splits=n_splits, n_repeats=n_repeats, verbose=verbose,
                              random_state=random_state, stratify=stratify)

    trial.set_user_attr('mse_sd', metrics['mse'][1])

    return metrics['mse'][0]
