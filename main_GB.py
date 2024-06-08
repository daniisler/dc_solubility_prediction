import os

from dotenv import load_dotenv
from gradient_boosting import gradient_boosting
from logger import logger

# Env
PROJECT_ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
logger = logger.getChild('main')

# Input data file
input_type = 'Aq'  # 'Aq' or 'Big'
input_data_filename = f'{input_type}SolDB_filtered_log.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)

# Where so save the best model weights and name of study
study_name = 'Aq_rd_tt_opt_lr'
model_save_folder = study_name
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models/gradient_boosting', model_save_folder)
output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')

# Select fingerprint for model
selected_fp = {'tt_fp': (2048, 4)}  # Possible values: 'm_fp': (2048, 2), 'rd_fp': (2048, (1, 7)), 'ap_fp': (2048, (1, 30)),
# 'tt_fp': (2048, 4)

# Select descriptors for model
descriptors = {
    # 'MolLogP': Descriptors.MolLogP,
    # 'LabuteASA': Descriptors.LabuteASA,
    # 'MolWt': Descriptors.MolWt
    # 'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO,
    # 'Kappa3': Descriptors.Kappa3,
    # 'PEOE_VSA2': Descriptors.PEOE_VSA2,
    # 'PEOE_VSA9': Descriptors.PEOE_VSA9
    # 'molecular_weight': Descriptors.MolWt,
    # 'TPSA': Descriptors.TPSA,
    # 'num_h_donors': Descriptors.NumHDonors,
    # 'num_h_acceptors': Descriptors.NumHAcceptors,
    # 'num_rotatable_bonds': Descriptors.NumRotatableBonds,
    # 'num_atoms': Chem.rdchem.Mol.GetNumAtoms,
    # 'num_heteroatoms': Descriptors.NumHeteroatoms,
    # 'num_valence_electrons': Descriptors.NumValenceElectrons,
    # 'num_rings': Descriptors.RingCount,
    # 'max_abs_partial_charge': Descriptors.MaxAbsPartialCharge,
    # 'max_partial_charge': Descriptors.MaxPartialCharge,
    # 'min_abs_partial_charge': Descriptors.MinAbsPartialCharge,
    # 'min_partial_charge': Descriptors.MinPartialCharge,
    # 'num_NHOH': Descriptors.NHOHCount,
    # 'fraction_C_sp3': Descriptors.FractionCSP3
}

# Select existing descriptors from data file.
descriptors_df_list = []

# Select list of solvents used in model
solvents = []
# df_temp = pd.read_csv(input_data_filepath)
# solvents = list(df_temp[df_temp['Solvent'].apply(lambda name: name.endswith('ol'))]['Solvent'].unique())
# solvents = ['methanol', 'ethanol', 'n-propablo', 'isopropanol']

# Select CV mode used (group k-fold for BigSolDB)
group_kfold = input_type == 'Big'

# Set parameters for CV
n_splits = 5
n_repeats = 1

# Random state for data splitting (only needed if group_kfold is False)
random_state = 0

# Choose max time for optimization in seconds
timeout = 3600

# Set parameters for lightgbm
lightgbm_params = {
    'num_leaves': (100, 600),  # Number of leaves in a tree
    'learning_rate': (0.005, 0.1),  # Learning rate
    'n_estimators': (700, 2000),  # Number of boosting rounds
    'max_depth': (10, 35),  # Maximum tree depth
    # 'min_child_samples': (5, 40),  # Minimum number of data in a leaf
    'subsample': (0.3, 1),  # Subsample ratio of the training data
    'colsample_bytree': (0.5, 1),  # Subsample ratio of columns when constructing each tree
    # 'extra_trees': [True, False],
}

# Settings for optuna.pruner
min_resource = 'auto'
reduction_factor = 2
min_early_stopping_rate = 0
bootstrap_count = 0

# Settings for optuna study creation
direction = 'minimize'  # 'maximize' if higher scores are desired (used for different validation score calculations)
storage = 'sqlite:///db.sqlite3'  # use optuna-dashboard sqlite:///db.sqlite3 to look at runs

# Choose if some results should be printed in console
verbose = True

gradient_boosting(
    input_data_filepath=input_data_filepath,
    output_paramoptim_path=output_paramoptim_path,
    model_save_dir=model_save_dir,
    study_name=study_name,
    selected_fp=selected_fp,
    descriptors=descriptors,
    descriptors_df_list=descriptors_df_list,
    solvents=solvents,
    lightgbm_params=lightgbm_params,
    group_kfold=group_kfold,
    n_splits=n_splits,
    n_repeats=n_repeats,
    timeout=timeout,
    random_state=random_state,
    min_resource=min_resource,
    reduction_factor=reduction_factor,
    min_early_stopping_rate=min_early_stopping_rate,
    bootstrap_count=bootstrap_count,
    direction=direction,
    storage=storage,
    verbose=verbose,
)
