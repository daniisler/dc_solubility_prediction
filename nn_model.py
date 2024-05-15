import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from logger import logger

# Env
logger = logger.getChild('nn_model')


class SolubilityModel(LightningModule):
    '''PyTorch model for solubility prediction based on molecular fingerprints.

    :param int input_size: size of the input data
    :param list n_neurons_hidden_layers: number of neurons in the hidden layers, one layer per element
    :param Dataset train_data: training dataset
    :param Dataset valid_data: validation dataset
    :param Dataset test_data: test dataset
    :param nn.Module activation_function: activation function
    :param int batch_size: batch size
    :param float lr: learning rate
    :param torch.optim optimizer: optimization algorithm
    :param torch.nn loss_function: loss function
    :param int num_workers: number of workers for data loading, set to 0 for Windows/MacOS

    :methods:
    training_step: define the training step
    validation_step: define the validation step
    test_step: define the test step
    configure_optimizers: configure the optimization algorithm
    forward: define the forward pass
    train_dataloader: prepare training batches
    val_dataloader: prepare validation batches
    test_dataloader: prepare testing batches

    :return: SolubilityModel object

    '''
    def __init__(self, input_size, n_neurons_hidden_layers, train_data, valid_data, test_data, activation_function=nn.ReLU, batch_size=254, lr=1e-3, optimizer=torch.optim.Adam, loss_function=nn.functional.mse_loss, num_workers=0):
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

    # pylint: disable=arguments-differ, unused-argument

    # Define the train step
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_function(z, y)
        self.log("Train loss", loss, on_epoch=True, on_step=False)
        return loss

    # Define the validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_function(z, y)
        self.log("Validation loss", loss)

    # Define the test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_function(z, y)
        self.log("Test loss", loss)

    # Configure the optimization algorithm
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    # Define the forward pass
    def forward(self, x):
        return self.model(x).flatten()

    # pylint: enable=arguments-differ, unused-argument

    # Prepare training batches
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # Prepare validation batches
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # Prepare testing batches
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # Initialize the weights and biases
    def init_weights(self, weight_init):
        '''Initialize the weights and biases of the model.

        :param str weight_init: weight initialization method, possible values: 'target_mean', 'default'

        '''
        # Initialize bias based on output normalization
        if weight_init == 'target_mean':
            # Compute mean and standard deviation of output values in the training data
            train_outputs = torch.cat([sample[1] for sample in self.train_data])
            target_mean = train_outputs.mean()

            # Initialize output layer bias to map inputs to desired output mean
            with torch.no_grad():
                self.model.output.bias.data.fill_(target_mean)

        # Default initialization (He)
        elif weight_init == 'default':
            pass
        else:
            logger.warning(f'Unknown weight initialization method {weight_init}, using default initialization...')
