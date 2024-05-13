import pandas as pd
import lightgbm
import os
import pandas as pd
import numpy as np
import optuna

from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
# from rdkit.Chem.PandasTools import AddMoleculeColumnToFrame
from rdkit.Chem import Descriptors
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, train_test_split, cross_val_predict, cross_val_score, RepeatedKFold
from collections import defaultdict
from tqdm import tqdm
from data_prep import gen_train_valid_test, calc_fingerprints
from sklearn.metrics import mean_squared_error
from logger import logger

# TODO: implement different fingerprints
# TODO: check with daniel if implementation of m_fp is correct
def gradient_boosting(input_data_filepath, output_paramoptim_path=None, model_save_dir=None, selected_fp={'m_fp': (2048, 2)}, descriptors=None, scale_transform=True, train_valid_test_split=None, n_splits=5, n_repeats=1, random_state=0, wandb_identifier='undef', wandb_mode='offline'):
    '''

    '''

    if train_valid_test_split is None:
        train_valid_test_split = [0.8, 0.1, 0.1]

    # TODO: decide which descriptors make sense from a chemists POV
    if descriptors is None:
        descriptors = {
            'molecular_weight': Descriptors.MolWt,
            'TPSA': Descriptors.TPSA,
            'num_h_donors': Descriptors.NumHDonors,
            'num_h_acceptors': Descriptors.NumHAcceptors,
            'num_rotatable_bonds': Descriptors.NumRotatableBonds,
            'num_atoms': Descriptors.HeavyAtomCount,
            'num_atoms_with_hydrogen': Descriptors.HeavyAtomCount,
            'num_atoms_without_hydrogen': Descriptors.HeavyAtomCount,
            'num_heteroatoms': Descriptors.NumHeteroatoms,
            'num_valence_electrons': Descriptors.NumValenceElectrons,
            'num_rings': Descriptors.RingCount,
        }

    # Check if the input file exists
    if not os.path.exists(input_data_filepath):
        raise FileNotFoundError(f'Input file {input_data_filepath} not found.')

    # read input data
    main_df = pd.read_csv(input_data_filepath)
    df = main_df[~(main_df['SMILES_Solvent'] == '-')]

    # calculate molecule object and fingerprints for solutes and solvents
    df = calc_fingerprints(df=df, selected_fp=selected_fp, solvent_fp=True)

    # calculate descriptors
    # TODO: logger should print all used descriptors
    logger.info(f'Calculating descriptors: ')
    desc_cols = []
    for col in ['mol', 'mol_solvent']:
        for desc_name, desc_func in descriptors.items():
            df[f"{col}_{desc_name}"] = df[col].apply(desc_func)
            desc_cols.append(f"{col}_{desc_name}")

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=selected_fp['m_fp'][0], radius=selected_fp['m_fp'][1])
    fp_cols = []

    # make a new column in df for every element in fingerprint list (easier format to handle)
    for col in ['mol', 'mol_solvent']:
        fingerprints = np.stack(df[col].apply(lambda x: np.array(fp_gen.GetFingerprint(x))).values)
        df[[f'{col}_fp_{i}' for i in range(fingerprints.shape[1])]] = fingerprints
        fp_cols.extend([f'{col}_fp_{i}' for i in range(fingerprints.shape[1])])

    target_col = 'Solubility'
    # TODO: implement temperature?
    feature_cols = fp_cols + desc_cols
    X = df[feature_cols].values
    y = df[target_col].values

    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', reduction_factor=2, min_early_stopping_rate=0, bootstrap_count=0)

    # Create a study object and optimize
    study = optuna.create_study(direction='minimize', pruner=pruner)  # You may need to adjust the direction based on your scoring method
    study.optimize(lambda trial: objective(trial, X=X, y=y, n_splits=n_splits, n_repeats=n_repeats), timeout=60)

    return 0

def cv_model_optuna(
    trial,
    model,
    X,
    y,
    metric_fs = None,
    return_metrics_list: bool = False,
    n_splits: int = 6,
    n_repeats: int = 4,
    random_state: int = 0,
    verbose: bool = False,
    ):
  """
  Perform RepeatedStratifiedKFold Cross Validation of a model with a sklearn API (fit, predict) and calculate specified metrics on the oof predictions.

  Additionally to cv_model this function reports intermediate values to an optuna pruner and prunes the trial if needed.
  """
  metric_fs = metric_fs if metric_fs is not None else {'mse': mean_squared_error}

  skf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
  folds = skf.split(X, y)

  metrics_list = defaultdict(list)
  for idx, (train_index, val_index) in tqdm(enumerate(folds)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model.fit(X_train, y_train.reshape(-1, 1))
    y_pred = model.predict(X_val)

    for metric_name, metric_f in metric_fs.items():
      metrics_list[metric_name].append(metric_f(y_val.ravel(), y_pred.ravel()))

    if verbose:
      print(f"mean mse: {np.mean(metrics_list['mse'])}")

    trial.report(np.mean(metrics_list['mse']), idx)
    if trial.should_prune():
      #print("TRIAL SHOULD BE PRUNED")
      raise optuna.TrialPruned()

  metrics = {k: (np.mean(v), np.std(v)) for k, v in metrics_list.items()}

  if not return_metrics_list:
    return metrics

  return metrics, metrics_list

def objective(trial, X, y, n_splits: int = 5, n_repeats: int = 4, verbose=True, lightgbm_params=None):
    '''

    '''
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

    #print("="*100) # TODO: add this to logger?

    # Sample hyperparameters
    params = {
        'num_leaves': trial.suggest_int('num_leaves', *lightgbm_params['num_leaves']),
        #'learning_rate': trial.suggest_loguniform('learning_rate', *lightgbm_params['learning_rate']),
        'n_estimators': trial.suggest_int('n_estimators', *lightgbm_params['n_estimators']),
        'max_depth': trial.suggest_int('max_depth', *lightgbm_params['max_depth']),
        #'min_child_samples': trial.suggest_int('min_child_samples', *lightgbm_params['min_child_samples']),
        'subsample': trial.suggest_float('subsample', *lightgbm_params['subsample']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', *lightgbm_params['colsample_bytree']),
        #'extra_trees': trial.suggest_categorical('extra_trees', lightgbm_params['extra_trees'])
    }

    # Create estimator
    model = LGBMRegressor(**params, verbose=-1)

    # Evaluate model
    metrics = cv_model_optuna(trial, model, X, y, n_splits=n_splits, n_repeats=n_repeats, verbose=verbose)

    return metrics['mse'][0]

