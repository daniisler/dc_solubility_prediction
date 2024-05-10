import os
import numpy as np
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
from logger import logger

# Env
logger = logger.getChild('data_prep')


# SolubilityDataset class
class SolubilityDataset(Dataset):

    '''Dataset class for solubility prediction.
    :param np.array X: input data
    :param np.array y: target data

    :methods:
    __len__: return the length of the dataset
    __X_size__: return the size of the input data
    __getitem__: return the input and target data at the given index

    :return: SolubilityDataset object

    '''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __X_size__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not torch.is_tensor(self.X):
            X_ = torch.as_tensor(self.X[idx].astype(np.float32))
        else:
            X_ = self.X[idx]
        if not torch.is_tensor(self.y):
            y_ = torch.as_tensor(self.y[idx].astype(np.float32).reshape(-1))
        else:
            y_ = self.y[idx]

        return X_, y_


# Filter for temperature, rounded to round_to decimal places
def filter_temperature(df, T, round_to=0):
    return df[round(df['T,K'] - T, round_to) == 0]


# Calculate the Morgan fingerprints
def calc_fingerprints(df, selected_fp, solvent_fp=False):
    '''Calculate the selected fingerprints for the molecules and the solvent SMILES.

    :param pd.DataFrame df: input dataframe
    :param dict selected_fp: dict of selected fingerprints, possible keys: 'm_fp', 'rd_fp', 'ap_fp', 'tt_fp'
    :param bool solvent: whether to calculate the solvent fingerprints
    :param list sizes: list of fingerprint sizes, always 4 elements
    :param list [float, tuple, tuple, int] radii: fingerprint radii respective parameters, always 4 elements

    :return: dataframe with calculated fingerprints

    '''
    selected_fp_keys = selected_fp.keys()
    logger.info(f'Calculating fingerprints: {selected_fp_keys}...')
    df['mol'] = df['SMILES'].apply(MolFromSmiles)
    if solvent_fp:
        df['mol_solvent'] = df['SMILES_Solvent'].apply(MolFromSmiles)
    if 'm_fp' in selected_fp_keys:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=selected_fp['m_fp'][0], fpSize=selected_fp['m_fp'][0])
        df['m_fp'] = df['mol'].apply(mfpgen.GetFingerprint)
        if solvent_fp:
            df['m_fp_solvent'] = df['mol_solvent'].apply(mfpgen.GetFingerprint)
    if 'rd_fp' in selected_fp_keys:
        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=selected_fp['rd_fp'][0], minPath=selected_fp['rd_fp'][1][0], maxPath=selected_fp['rd_fp'][1][1])
        df['rd_fp'] = df['mol'].apply(rdkgen.GetFingerprint)
        if solvent_fp:
            df['rd_fp_solvent'] = df['mol_solvent'].apply(rdkgen.GetFingerprint)
    if 'ap_fp' in selected_fp_keys:
        apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=selected_fp['ap_fp'][0], min_distance=selected_fp['ap_fp'][1][0], max_distance=selected_fp['ap_fp'][1][1])
        df['ap_fp'] = df['mol'].apply(apgen.GetFingerprint)
        if solvent_fp:
            df['ap_fp_solvent'] = df['mol_solvent'].apply(apgen.GetFingerprint)
    if 'tt_fp' in selected_fp_keys:
        ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=selected_fp['tt_fp'][0], torsionAtomCount=selected_fp['tt_fp'][1])
        df['tt_fp'] = df['mol'].apply(ttgen.GetFingerprint)
        if solvent_fp:
            df['tt_fp_solvent'] = df['mol_solvent'].apply(ttgen.GetFingerprint)

    return df


def gen_train_valid_test(X, y, split, scale_transform, model_save_dir, random_state):
    '''Separate the data according to a split[0]split[1]/split[2] train/validation/test split.

    :param np.array X: input data
    :param np.array y: target data
    :param list split: list of split ratios for train, validation, test datasets
    :param bool scale_transform: whether to scale the data
    :param int random_state: random state for data splitting for reproducibility

    :return: train, validation and test datasets

    '''
    if not np.sum(split) == 1.0:
        raise ValueError('The sum of the split ratios must be 1.')

    logger.info('Generating train, validation and test datasets...')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=split[1]+split[2], random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=split[2], random_state=random_state)

    # Data normalization
    logger.info('Normalizing data...')
    if scale_transform:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        # Save the scaler
        with open(os.path.join(model_save_dir, 'scaler.pkl'), 'wb') as f:
            dump(scaler, f)

    # Create SolubilityDataset objects
    train_data = SolubilityDataset(X_train, y_train)
    valid_data = SolubilityDataset(X_valid, y_valid)
    test_data = SolubilityDataset(X_test, y_test)

    return train_data, valid_data, test_data
