# dc_solubility_prediction

A Project for the digital chemistry course FS24: Predicting the Solubility of Organic Molecules in Organic Solvents and Water.

## Data Curation

## Descriptor Calculation

## Filtration and Preprocessing

Before feeding the data to the neural network, it was filtered for a single temperature and solvent. Afterwards, fingerprints are calculated. The implemented options are Morgan fingerprints, RDkit fingerprints, atomic pair fingerprints and topological torsion fingerprints, all calculated using the `rdkit` library. The input for the model is prepared by concatenation of the selected fingerprints and subsequent normalization by a standard scaler. The data is split into training, validation, and test sets (80/10/10 default). The data preparation is done in `data_prep.py`.

## Hyperparameter Optimization

The hyperparameter optimization is done using pytorch-lightning and can be tracked with W&B. In order to use W&B, you need to create an account at [wandb.ai](https://wandb.ai/) and paste your API key in the `.env` file (create the file) in the root directory of the project. Copy the `.env.template` file and paste your API key in the `WANDB_API_KEY` variable. The use of W&B is not necessary, and can be disabled by setting the `wandb_mode='disabled'`. Though it can help to track the progress of the optimization and better understand what is actually happening.

The optimization is performed on the validation dataset and the test data should remain untouched. The hyperparameters are saved in the file specified at the beginning of the optimization in `main.py`. The source code for the optimization can be found in `hyperparam_optim.py`, where a grid search over the provided hyperparameters is performed.

To play with the optimization parameters, simply change the variables at the beginning of the file `main.py` and check the logs in the W&B dashboard or the logs in the `logs` directory.

## Prediction

The prediction can also be run from `main.py`. To use an already trained model, set the `prediction_only` variable to `True` and specify the path to the model in the `model_save_folder` variable. Then paste the SMILES string of the molecule you want to predict in the `smiles` variable (towards the end of the file). The prediction will be printed in the console and written to the log file.

## TODO

- [ ] Optimize hyperparameters for the AqSolDB dataset and do a first evaluation on how well it performs.
- [ ] Implement temperature as a parameter. The data is currently filtered for a single temperature, but the model should be able to predict the solubility at different temperatures.
- [ ] Implement the solvent as a parameter. The data is currently filtered for a single solvent, but the model should be able to predict the solubility in different solvents.
- [x] Add a script that trains a model for the most common solvents and predicts the solubility of a molecule in all of them/a selected one of them.
- [ ] Evaluate performance of the different approaches.
- [ ] Apply some kind of delta learning from the descriptor calculation, e.g. by adding the dipole moment as input, which could be an important parameter. Compare the performance of the model with and without the dipole moment.
- [x] Find a better way to store the best hyperparameters of an optimization that allows to easily load them for a prediction.
