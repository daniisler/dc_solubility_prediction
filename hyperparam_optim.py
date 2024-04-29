import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import itertools

from nn_model import SolubilityModel
from logger import logger

# Env
logger = logger.getChild('hyperparam_optimization')

# EXAMPLE PARAM_GRID
# param_grid = {
#     'batch_size': [16, 32, 64],
#     'learning_rate': [0.0005, 0.001, 0.005],
#     'n_neurons_hidden_layers': [[16], [32], [64], [128], [256], [32, 16], [64, 32]],
#     'max_epochs': [50]
# }

def hyperparam_optimization(param_grid, train_data, valid_data, test_data, wandb_identifier='undef', wandb_disabled=None, early_stopping=True, ES_mode='min', ES_patience=5, ES_min_delta=0.05):
    best_score = np.inf
    best_hyperparams = {}
    # Test all possible combinations of hyperparameters
    combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    for combination in combinations:
        logger.info(f"\n*** Run with hyperparameters: {combination} ***\n")
        # Start W&B
        wandb.init(project=wandb_identifier, config=combination, mode=wandb_disabled)
        wandb_logger = WandbLogger()
        # Create an instance of our neural network
        nn_model = SolubilityModel(
            input_size=train_data.__X_size__(),
            n_neurons_hidden_layers=combination['n_neurons_hidden_layers'],
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            lr=combination['learning_rate'],
            batch_size=combination['batch_size']
        )
        # Reset the early stopping callback
        if early_stopping:
            early_stop_callback = EarlyStopping(monitor="Validation loss", min_delta=ES_min_delta, patience=ES_patience, verbose=False, mode=ES_mode)
        # Define trainer
        trainer = Trainer(
            max_epochs=combination['max_epochs'],
            logger=wandb_logger,
            callbacks=[early_stop_callback],
            accelerator="gpu" if torch.cuda.is_available() else "cpu" # use GPU if available
        )
        # Train the model
        trainer.fit(model=nn_model)
        # Validate the model
        val_loss = trainer.validate(model=nn_model)[0]['Validation loss']
        # Update the best score and hyperparameters if current model is better
        if val_loss < best_score:
            best_score = val_loss
            best_hyperparams = {
                'hidden_size': combination['n_neurons_hidden_layers'],
                'learning_rate': combination['learning_rate'],
                'batch_size': combination['batch_size'],
                'n_epochs': trainer.current_epoch
            }
        wandb.finish()

    return best_hyperparams, best_score
