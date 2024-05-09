import pandas as pd
import numpy as np
import os
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')

# Filter duplicates and take the first value (for now) TODO: improve me
df = pd.read_csv(os.path.join(DATA_DIR, 'BigSolDB.csv'))

# Standardize smiles
df['SMILES'] = df['SMILES'].apply(StandardizeSmiles)

# Duplicates means duplicates in all of the columns 'SMILES', 'Solvent', 'T,K'
df.drop_duplicates(subset=['SMILES', 'Solvent', 'T,K'], keep='first', inplace=True)

# Apply log transformation to the solubility
df['Solubility'] = df['Solubility'].apply(lambda x: np.log10(x))

df.to_csv(os.path.join(DATA_DIR, 'BigSolDB_filtered_log.csv'), index=False)

# Bring the aqsoldata to the same format as BigSolDB
df_aqsol = pd.read_csv(os.path.join(DATA_DIR, 'combined_cleaned_data.csv'))

print(df_aqsol.columns.values)
print(df_aqsol.shape)

# Standardize smiles
df_aqsol['SMILES'] = df_aqsol['smiles'].apply(StandardizeSmiles)
df_aqsol.drop(columns=['smiles'], inplace=True)
df_aqsol['Solvent'] = 'water'
df_aqsol['T,K'] = 298.15
df_aqsol['Source'] = 'AqSolDB'
df_aqsol['Solubility'] = df_aqsol['logS']
df_aqsol['SMILES_Solvent'] = 'O'
df_aqsol.drop(columns=['logS'], inplace=True)

print(df_aqsol.columns.values)
# Drop duplicate smiles
df_aqsol.drop_duplicates(subset=['SMILES', 'Solvent', 'T,K'], keep='first', inplace=True)
print(df_aqsol.shape)
df_aqsol.to_csv(os.path.join(DATA_DIR, 'AqSolDB_filtered_log.csv'), index=False)
