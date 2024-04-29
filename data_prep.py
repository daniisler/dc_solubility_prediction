import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger import logger

# Env
logger = logger.getChild('data_prep')

# SolubilityDataset class
class SolubilityDataset(Dataset):

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
def filter_temperature(df, T=293, round_to=0):
    return df[round(df['T,K'] - T, round_to) == 0]

# Calculate the Morgan fingerprints
def calc_fingerprints(df, size=2048, radius=2):
    logger.info('Setting up Fingerprint generators...')
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=size)
    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=size)
    apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=size)
    ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=size)

    logger.info('Calculating fingerprints for molecules SMILES...')
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df['m_fp'] = df['mol'].apply(mfpgen.GetFingerprint)
    df['rd_fp'] = df['mol'].apply(rdkgen.GetFingerprint)
    df['ap_fp'] = df['mol'].apply(apgen.GetFingerprint)
    df['tt_fp'] = df['mol'].apply(ttgen.GetFingerprint)

    logger.info('Calculating fingerprints for solvent SMILES...')
    df['mol_solvent'] = df['SMILES_Solvent'].apply(Chem.MolFromSmiles)
    df['m_fp_solvent'] = df['mol_solvent'].apply(mfpgen.GetFingerprint)
    df['rd_fp_solvent'] = df['mol_solvent'].apply(rdkgen.GetFingerprint)
    df['ap_fp_solvent'] = df['mol_solvent'].apply(apgen.GetFingerprint)
    df['tt_fp_solvent'] = df['mol_solvent'].apply(ttgen.GetFingerprint)

    return df

# Separate the data according to a 80/10/10 train/validation/test split
def gen_train_valid_test(X, y, scale_transform=True, random_state=0):
    logger.info('Generating train, validation and test datasets...')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Data normalization
    logger.info('Normalizing data...')
    if scale_transform:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    # Create SolubilityDataset objects
    train_data = SolubilityDataset(X_train, y_train)
    valid_data = SolubilityDataset(X_valid, y_valid)
    test_data = SolubilityDataset(X_test, y_test)
    
    return train_data, valid_data, test_data