# dc_solubility_prediction

A Project for the digital chemistry course FS24: Predicting the Solubility of Organic Molecules in Organic Solvents and Water.

## Data Curation

## Descriptor Calculation

## Filtration and Preprocessing

Before feeding the data to the neural network, it was filtered for a single temperature and solvent. The data was preprocessed by generating molecular fingerprints as descriptors, before splitting the data into training, validation, and test sets (80/10/10). For preprocessing, a standard scaler was used to normalize the data. These steps are performed in `data_prep.py`.

## Hyperparameter Optimization

The hyperparameter optimization is done using pytorch-lightning and tracked with W&B. In order to use W&B, you need to create an account at [wandb.ai](https://wandb.ai/) and paste your API key in the (you should create it) `.env` file in the root directory of the project. Copy the `.env.template` file and paste your API key in the `WANDB_API_KEY` variable. The use of W&B is not necessary, and can be disabled by setting the `wandb_mode='disabled'`. Though it can help to track the progress of the optimization and better understand what is actually happening.

The optimization is performed on the validation dataset and the test data should remain untouched. The hyperparameters are saved in the file specified at the beginning of the optimization in `main.py`. The source code for the optimization can be found in `hyperparam_optim.py`, where a grid search over the provided hyperparameters is performed.

To play with the optimization parameters, simply change the variables at the beginning of the file `main.py` and check the logs in the W&B dashboard or the logs in the `logs` directory.
