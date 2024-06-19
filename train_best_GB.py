import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data_prep import calc_fingerprints
from logger import logger

# A short optimization with main_GB.py gave the following results:
# Trial finished with mean mse: 0.791816856511635, mse_std: 0.07940024576960349 and parameters: {'num_leaves': 425, 'learning_rate': 0.009414773107327284, 'n_estimators': 1059, 'max_depth': 32, 'subsample': 0.6337082549157453, 'colsample_bytree': 0.5037365608016737}
# With the settings as listed below


# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
logger = logger.getChild('main')

# Input data file
input_type = 'Big'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_log.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)

# Where so save the best model weights and name of study
study_name = 'Aq_rd_tt_opt_lr'
continue_study = False
model_save_folder = study_name
model_save_dir = os.path.join(PROJECT_ROOT) #, 'saved_models/gradient_boosting', model_save_folder)
os.makedirs(model_save_dir, exist_ok=True)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')

# Select fingerprint for model
selected_fp = {'ap_fp': (2048, (1, 30))}  # Possible values: 'm_fp': (2048, 2), 'rd_fp': (2048, (1, 7)), 'ap_fp': (2048, (1, 30)),
# 'tt_fp': (2048, 4)

# Select descriptors for model
descriptors = {
    'MolLogP': Descriptors.MolLogP,
    'LabuteASA': Descriptors.LabuteASA,
    'MolWt': Descriptors.MolWt,
    'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO,
    'Kappa3': Descriptors.Kappa3,
    'PEOE_VSA2': Descriptors.PEOE_VSA2,
    'PEOE_VSA9': Descriptors.PEOE_VSA9,
    'molecular_weight': Descriptors.MolWt,
    'TPSA': Descriptors.TPSA,
    'num_h_donors': Descriptors.NumHDonors,
    'num_h_acceptors': Descriptors.NumHAcceptors,
    'num_rotatable_bonds': Descriptors.NumRotatableBonds,
    'num_atoms': Chem.rdchem.Mol.GetNumAtoms,
    'num_heteroatoms': Descriptors.NumHeteroatoms,
    'num_valence_electrons': Descriptors.NumValenceElectrons,
    'num_rings': Descriptors.RingCount,
    'max_abs_partial_charge': Descriptors.MaxAbsPartialCharge,
    'max_partial_charge': Descriptors.MaxPartialCharge,
    'min_abs_partial_charge': Descriptors.MinAbsPartialCharge,
    'min_partial_charge': Descriptors.MinPartialCharge,
    'num_NHOH': Descriptors.NHOHCount,
    'fraction_C_sp3': Descriptors.FractionCSP3
}

# Select existing descriptors from data file that were calculated with descriptors_calculation.py
descriptors_df_list = []

# Apply a standard scaler to the input data
scale_transform = False

# Select list of solvents used in model
solvents = ['water']

# Select CV mode used (group k-fold for BigSolDB)
group_kfold = input_type == 'Big'

train_best_model = True

# Set parameters for CV
n_splits = 5
n_repeats = 1

# Random state for data splitting (only needed if group_kfold is False)
random_state = 0

# Set parameters for lightgbm
lightgbm_params = {'num_leaves': 425, 'learning_rate': 0.009414773107327284, 'n_estimators': 1059, 'max_depth': 32, 'subsample': 0.6337082549157453, 'colsample_bytree': 0.5037365608016737}

# Settings for optuna.pruner
min_resource = 'auto'
reduction_factor = 2
min_early_stopping_rate = 0
bootstrap_count = 0

# Check if the input file exists
if not os.path.exists(input_data_filepath):
    raise FileNotFoundError(f"Input file {input_data_filepath} not found.")

# Read input data (and filter lines with '-' as SMILES_Solvent)
df = pd.read_csv(input_data_filepath)
df = df[df["SMILES_Solvent"] != "-"]

if solvents:
    df = df[df["Solvent"].isin(solvents)]
    logger.info(f"Solvents used for filtering: {solvents}")

logger.info(f"Length of Data: {df.shape[0]}")

# Calculate molecule object and fingerprints for solutes and solvents and rename column
df = calc_fingerprints(df=df, selected_fp=selected_fp, solvent_fp=True)
df.rename(
    columns={list(selected_fp.keys())[0]: f"{list(selected_fp.keys())[0]}_mol"},
    inplace=True,
)

# Calculate descriptors
desc_cols = []
if descriptors:
    logger.info(f"Calculating descriptors:{list(descriptors.keys())}")
    for col in ["mol", "mol_solvent"]:
        for desc_name, desc_func in descriptors.items():
            df[f"{col}_{desc_name}"] = df[col].apply(desc_func)
            desc_cols.append(f"{col}_{desc_name}")

    logger.info(f"Descriptors used as features: {desc_cols}")

# Add existing descriptors from DataFrame
if descriptors_df_list:
    logger.info(f"Adding existing descriptor columns:{descriptors_df_list}")
    for desc_df_name in descriptors_df_list:
        if desc_df_name in df.columns:
            desc_cols.append(desc_df_name)
        else:
            logger.warning(
                f"Descriptor {desc_df_name} not found in DataFrame columns"
            )
else:
    logger.info(
        "No descriptors from the DataFrame were added: descriptors_df_list is not defined or is empty"
    )

logger.info(f"Final descriptors used as features: {desc_cols}")

# Make a new column in df for every element in fingerprint list (easier format to handle)
mol_fingerprints = np.stack(df[f"{list(selected_fp.keys())[0]}_mol"].values)
df_mol_fingerprints = pd.DataFrame(
    mol_fingerprints,
    columns=[f"mol_fp_{i}" for i in range(mol_fingerprints.shape[1])],
)

solvent_fingerprints = np.stack(df[f"{list(selected_fp.keys())[0]}_solvent"].values)
df_solvent_fingerprints = pd.DataFrame(
    mol_fingerprints,
    columns=[f"solvent_fp_{i}" for i in range(solvent_fingerprints.shape[1])],
)

# Reset indices of df so pd.concat does not produce any NaN
df.reset_index(drop=True, inplace=True)
df = pd.concat([df, df_mol_fingerprints, df_solvent_fingerprints], axis=1)

# Create list of feature and target columns
fp_cols = list(df_mol_fingerprints.columns) + list(df_solvent_fingerprints.columns)
target_col = "Solubility"
feature_cols = fp_cols + desc_cols

if train_best_model:

    # Add temperature as feature, if group k-fold CV is used
    if group_kfold:
        feature_cols = feature_cols + ["T,K"]
    metric_fs = {"mse": mean_squared_error}

    model = LGBMRegressor(
        **lightgbm_params, verbose=-1
    )
    # Use GroupKFold or RepeatedKFold for CV
    if group_kfold:
        # Use SMILES column as variable used for grouping
        groups = df["SMILES"]

        # Convert pandas dfs to numpy.ndarrays for better performance, consider using .to_numpy instead
        X = df[feature_cols].values
        y = df[target_col].values

        gkf = GroupKFold(n_splits=n_splits)
        folds = gkf.split(X, y, groups=groups)

    else:
        # Convert pandas dfs to numpy.ndarrays for better performance, consider using .to_numpy instead
        X = df[feature_cols].values
        y = df[target_col].values

        skf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        folds = skf.split(X, y)

    metrics_list = defaultdict(list)
    # loop over different folds (tqdm adds progress bar to loop)
    for idx, (train_index, val_index) in tqdm(enumerate(folds)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        # Scale the data
        if scale_transform:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_val)
        for metric_name, metric_f in metric_fs.items():
            metrics_list[metric_name].append(metric_f(y_val.ravel(), y_pred.ravel()))

        # Save best model
        if metrics_list['mse'][idx] <= min(metrics_list['mse']):
            joblib.dump(model, os.path.join(model_save_dir, f"model_GB.pkl"))
            joblib.dump(scaler, os.path.join(model_save_dir, f"scaler_GB.pkl"))

    print(f"mean mse: {np.mean(metrics_list['mse'])}")
    print(f"model with lowest mse: {np.argmin(metrics_list['mse'])}")

# Load model
model = joblib.load(os.path.join(model_save_dir, f"model_GB.pkl"))
scaler = joblib.load(os.path.join(model_save_dir, f"scaler_GB.pkl"))

# Predict
df[feature_cols] = scaler.transform(df[feature_cols])
df['Solubility_pred'] = model.predict(df[feature_cols])

# Calculate MSE
mse = mean_squared_error(df['Solubility'], df['Solubility_pred'])
logger.info(f"Mean squared error: {mse}")