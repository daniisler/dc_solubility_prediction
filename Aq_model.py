# Used Data table with several descriptors and make a multivariate linear regression model: 
# use exersices 5-7

# Import dependencies
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#from dotenv import load_dotenv#not dure if this is needed
from logger import logger
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import r2_score

# Input data file
input_type = 'Aq'  # 'Aq' the only available so far
input_data_filename = f'{input_type}SolDB_filtered_log.csv'
input_data_filepath = os.path.join(DATA_DIR, input_data_filename)

# Filter for solvents #TODO(list) A separate model is trained for each solvent in the list
solvents = ['water']  # ['methanol', 'ethanol', 'water', 'toluene', 'chloroform', 'benzene', 'acetone']
# Filter for temperature in Kelvin; None for no filtering
T = 298
# Where to save the model
model_save_folder = 'AqSolDB_mvlm'
model_save_dir = os.path.join(PROJECT_ROOT, 'saved_models', model_save_folder)
#TODO not needed ? : #output_paramoptim_path = os.path.join(model_save_dir, 'hyperparam_optimization.json')

# Scale the input data
scale_transform = True
# Train/validation/test split
train_valid_test_split = [0.8, 0.1, 0.1]
# Random state for data splitting
random_state = 0
# Wandb identifier
wandb_identifier = 'AqSolDB_mvlm'
wandb_mode = 'online'

