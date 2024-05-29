import os

import joblib
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from logger import logger
from data_prep import calc_fingerprints
from filtration_functions import filter_solvent

from matplotlib import rc, rcParams, cm
# global settings of plots DO NOT CHANGE
medium_fontsize = 20.5
rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\boldmath
"""
font = {'size': medium_fontsize, 'family': 'sans-serif', 'weight': 'bold'}
rc('font', **font)
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['lines.linewidth'] = 2.5
rcParams['figure.figsize'] = (8, 8)
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.labelsize'] = medium_fontsize
rc('text', usetex=False)


def train_GB_model(
        input_data_filepath,
        output_data_filepath,
        best_params,
        selected_fp,
        descriptors,
        solvents,
        descriptors_df_list=None,
        group_kfold: bool = False,
):
    """
    Retrain GB model on whole dataset using the determined optimized hyperparameters and save X, y, y_pred in a .pkl file
    :param str input_data_filepath: path to the input data csv file
    :param str output_data_filepath: path to the output .pkl file where X, y, y_pred are saved
    :param dict best_params: determined dictionary of the best hyperparameters
    :param dict of tuples selected_fp: selected fingerprint for the model, possible keys:
        - m_fp: Morgan fingerprint, tuple of (size, radius)
        - rd_fp: RDKit fingerprint, tuple of (size, (minPath, maxPath))
        - ap_fp: Atom pair fingerprint, tuple of (size, (min_distance, max_distance))
        - tt_fp: Topological torsion fingerprint, tuple of (size, torsionAtomCount)
    :param dict descriptors: selected descriptors for the model
    :param list solvents: names of solvents which will be used to filter data (if list is empty, all solvents are used)
    :param list descriptors_df_list: selected existing descriptors within data file
    :param bool group_kfold: decides if group k-fold CV or normal k-fold CV is used (Smiles is variable used for grouping)
    :return: list of predicted solubility values
    """

    if solvents is None:
        solvents = []

    # Check if the input file exists
    if not os.path.exists(input_data_filepath):
        raise FileNotFoundError(f'Input file {input_data_filepath} not found.')

    # Read input data (and filter lines with '-' as SMILES_Solvent)
    df = pd.read_csv(input_data_filepath)
    df = df[df['SMILES_Solvent'] != '-']

    if solvents:
        df = filter_solvent(df, solvents=solvents)
        logger.info(f'Solvents used for filtering: {solvents}')

    logger.info(f'Length of Data: {df.shape[0]}')

    # Calculate molecule object and fingerprints for solutes and solvents and rename column
    df = calc_fingerprints(df=df, selected_fp=selected_fp, solvent_fp=True)
    df.rename(columns={list(selected_fp.keys())[0]: f'{list(selected_fp.keys())[0]}_mol'}, inplace=True)

    # Calculate descriptors
    desc_cols = []
    if descriptors:
        logger.info(f'Calculating descriptors:{list(descriptors.keys())}')
        for col in ['mol', 'mol_solvent']:
            for desc_name, desc_func in descriptors.items():
                df[f"{col}_{desc_name}"] = df[col].apply(desc_func)
                desc_cols.append(f"{col}_{desc_name}")

        logger.info(f'Descriptors used as features: {desc_cols}')

    # Add existing descriptors from DataFrame
    if descriptors_df_list:
        logger.info(f'Adding existing descriptor columns:{descriptors_df_list}')
        for desc_df_name in descriptors_df_list:
            if desc_df_name in df.columns:
                desc_cols.append(desc_df_name)
            else:
                logger.warning(f'Descriptor {desc_df_name} not found in DataFrame columns')
    else:
        logger.info('No descriptors from the DataFrame were added: descriptors_df_list is not defined or is empty')

    logger.info(f'Final descriptors used as features: {desc_cols}')

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

    # Add temperature as feature, if group k-fold CV is used
    if group_kfold:
        feature_cols = feature_cols + ['T,K']

    X = df[feature_cols].values
    y = df[target_col].values

    # Initialize the model
    model = lgb.LGBMRegressor(**best_params)

    # Train the model on the entire dataset
    model.fit(X, y)

    # Get cross-validated predictions
    y_pred = model.predict(X)

    joblib.dump([X, y, y_pred], f'{output_data_filepath}.pkl')

    logger.info(f'Model saved in {output_data_filepath}')

    return y_pred


def make_plots(model_file, saving_dir, saving_name):
    """
    Get model performance plots
    :param str model_file: path to file with X, y, y_pred values
    :param str saving_dir: path to directory where plots should be saved
    :param str saving_name: name that should be used to save plots (endings are plot specific)
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Input file {model_file} not found.')

    X, y, y_pred = joblib.load(model_file)

    os.chdir(saving_dir)

    # Scatter plot of true vs. predicted values
    plt.plot(y, y_pred, 'o')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.tight_layout()
    plt.savefig(f'{saving_name}_TV_vs_PV.png')
    plt.show()

    # Residual plot
    residuals = y - y_pred
    plt.plot(y, residuals, 'o')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.xticks(np.arange(-14, 6, 2))
    plt.tight_layout()
    plt.savefig(f'{saving_name}_R_vs_TV.png')
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(8, 8))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Distribution of Residuals', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'{saving_name}_Dist_of_R.png')
    plt.show()

