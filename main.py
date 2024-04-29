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

# Neural network model
class SolubilityModel(LightningModule):
    def __init__(self, input_size, n_neurons_hidden_layers, train_data, valid_data, test_data, activation_function=nn.ReLU, batch_size=254, lr=1e-3, optimizer=torch.optim.Adam, loss_function=F.mse_loss, num_workers=8):
        super().__init__()
        # Define model parameters
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.num_workers = num_workers
        # Define the training parameters
        self.lr = lr
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size

        # Define a sequential model
        self.model = nn.Sequential()
        if len(n_neurons_hidden_layers) == 0:
            self.model.add_module("input", nn.Linear(input_size, 1))
        else:
            self.model.add_module("input", nn.Linear(input_size, n_neurons_hidden_layers[0]))
            for i, n in enumerate(n_neurons_hidden_layers[:-1]):
                self.model.add_module(f"hidden_{i}", nn.Linear(n, n_neurons_hidden_layers[i+1]))
                self.model.add_module(f"activation_{i}", activation_function())
            self.model.add_module("output", nn.Linear(n_neurons_hidden_layers[-1], 1))

    # Define the train step
    # Mean square error as loss function
    def training_step(self, batch):
        x, y = batch
        z = self.model(x)
        loss = F.mse_loss(z, y)
        self.log("Train loss", loss)
        return loss

    # Define the validation step
    def validation_step(self, batch):
        x, y = batch
        z = self.model(x)
        loss = F.mse_loss(z, y)
        self.log("Validation loss", loss)

    # Define the test step
    def test_step(self, batch):
        x, y = batch
        z = self.model(x)
        loss = self.loss_function(z, y)
        self.log("Test loss", loss)

    # Configure the optimization algorithm
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(),lr=self.lr)
        return optimizer

    # Define the forward pass
    def forward(self, x):
        return self.model(x).flatten()

    # Prepare training batches
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # Prepare validation batches
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # Prepare testing batches
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

