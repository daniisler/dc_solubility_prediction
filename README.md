# dc_solubility_prediction

A Project for the digital chemistry course FS24: Predicting the Solubility of Organic Molecules in Organic Solvents and Water.

## Data Curation

## Descriptor Calculation

## Filtration and Preprocessing

Before feeding the data to the neural network, it was filtered for a single temperature and solvent. Afterwards, fingerprints are calculated. The implemented options are Morgan fingerprints, RDkit fingerprints, atomic pair fingerprints and topological torsion fingerprints, all calculated using the `rdkit` library. The input for the model is prepared by concatenation of the selected fingerprints and subsequent normalization by a standard scaler. The data is split into training, validation, and test sets (80/10/10 default). The data preparation is done in `data_prep.py`.

## Rdkit Descriptors

Optional selection of RDKit descriptors for model training (if `use_rdkit_descriptors == True`). List of possible descriptors: 'display(Descriptors._descList)'. Assign list of selected descriptors to `descriptors_list`.

## Hyperparameter Optimization

The hyperparameter optimization is done using pytorch-lightning and can be tracked with W&B. In order to use W&B, you need to create an account at [wandb.ai](https://wandb.ai/) and paste your API key in the `.env` file (create the file) in the root directory of the project. Copy the `.env.template` file and paste your API key in the `WANDB_API_KEY` variable. To use multiprocessing, you can also set the number of workers you would like to use for the data loaders. Note however that to run on euler this is specified in the deploy script `main_euler.sh`. The use of W&B is not necessary, and can be disabled by setting the `wandb_mode='disabled'`. Though it can help to track the progress of the optimization and better understand what is actually happening and is thus recommended.

The optimization is performed on the validation dataset and the test data should remain untouched. The hyperparameters are saved in the file specified at the beginning of the optimization in `main.py`. The source code for the optimization can be found in `hyperparam_optim.py`, where a grid search over the provided hyperparameters is performed.

To play with the optimization parameters, simply change the variables at the beginning of the file `main.py` and check the logs in the W&B dashboard or the logs in the `logs` directory. A grid search over the hyperparameters defined in the variable `param_grid` is performed -> note that this takes as many iteration as the product of the lengths of the lists in the dictionary values. The parameters set beforehand are fixed for the optimization and can be changed upon starting different optimization runs. The results of the optimization are saved in `<model_save_folder>/hyperparam_optimization_<solvent>.json` along with the best model weights (`weights_<solvent>.pth`), the best hyperparameters (`params_<solvent>.pkl`) and the scaler used for normalization (`scaler_<solvent>.pkl`). In order to train a model with specific hyperparameters and not do an optimization, just define the dictionary `params_grid` with lists that contain only the desired element and only one model will be trained.

## Prediction

The prediction can also be run from `main.py`. To use an already trained model, set the `prediction_only` variable to `True` and specify the path to the model(s) in the `model_save_folder` variable (of course a model needs to have been trained to perform a prediction.). Then paste the SMILES string of the molecule you want to predict in the `smiles` variable (towards the end of the file). Run `main.py` and the predicted solubility of the molecule in the specified solvents will be printed to the console and logged to the log file `logs/logging.log`.

## TODO

- [ ] Optimize hyperparameters for the AqSolDB dataset and do a first evaluation on how well it performs.
- [ ] Implement temperature as a parameter. The data is currently filtered for a single temperature, but the model should be able to predict the solubility at different temperatures. -> Note that for this a scaffold split is necessary, as the temperature is not a property of the molecule, but of the environment!
- [ ] Implement the solvent as a parameter. The data is currently filtered for a single solvent, but the model should be able to predict the solubility in different solvents. -> Also here a scaffold split will be necessary.
- [x] Add a script that trains a model for the most common solvents and predicts the solubility of a molecule in all of them/a selected one of them.
- [ ] Evaluate performance of the different approaches (on the test set) and compare them. Make nice plots of what works and hypothesize why.
- [ ] Apply some kind of delta learning from the descriptor calculation, e.g. by adding the dipole moment as input, which could be an important parameter. Compare the performance of the model with and without the dipole moment.
- [x] Find a better way to store the best hyperparameters of an optimization that allows to easily load them for a prediction.
- [x] Implement restore best weights for the model after ES
- [x] Consider to save the fingerprints instead of recalculating them every time
- [x] Implement a weight initialization for the model that is not random, but based on standard deviations and means of the training targets.
- [x] Implement learning rate scheduler
