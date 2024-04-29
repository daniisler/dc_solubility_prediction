# This is the main script for the project. It combines all the components

# TODO: Order us
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, LightningModule
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import itertools
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from logger import logger
from data_prep import gen_train_valid_test, filter_temperature, calc_fingerprints

# Env
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
logger = logger.getChild('main')

# Load the (filtered) data from csv
# COLUMNS: SMILES,"T,K",Solubility,Solvent,SMILES_Solvent,Source
df = pd.read_csv(os.path.join(DATA_DIR, 'BigSolDB_filtered.csv'))

# Filter for room temperature
df = filter_temperature(df)

# Calculate the fingerprints
df = calc_fingerprints(df)

# Define the input and target data
X = torch.tensor(df['m_fp'].values.tolist(), dtype=torch.float32)
y = torch.tensor(df['Solubility'].values, dtype=torch.float32).reshape(-1, 1)

# Split the data into train, validation and test set
train_dataset, valid_dataset, test_dataset = gen_train_valid_test(X, y)
