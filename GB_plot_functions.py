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
    if not os.path.exists(model_file):
        raise FileNotFoundError(f'Input file {model_file} not found.')

    X, y, y_pred = joblib.load(model_file)

    os.chdir(saving_dir)

    # Scatter plot of true vs. predicted values
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y, y=y_pred)
    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)
    plt.title('True vs Predicted Values', fontsize=18)
    max_val = max(max(y), max(y_pred))
    min_val = min(min(y), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'{saving_name}_TV_vs_PV.png')
    plt.show()

    # Residual plot
    residuals = y - y_pred
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Residuals', fontsize=16)
    plt.title('Residuals vs True Values', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
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

    # # Joint plot of true vs. predicted values
    # plt.figure(figsize=(10, 6))
    # sns.jointplot(x=y, y=y_pred, kind='scatter', marginal_kws=dict(bins=30, fill=True))
    # plt.xlabel('True Values', fontsize=16)
    # plt.ylabel('Predicted Values', fontsize=16)
    # plt.suptitle('True vs Predicted Values', y=0.98, fontsize=18)
    # plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15)
    # plt.savefig(f'{saving_name}_TV_vs_PV_with_dist.png')
    # plt.show()

