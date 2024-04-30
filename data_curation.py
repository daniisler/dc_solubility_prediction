import pandas as pd
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
print(df.shape)

df.to_csv(os.path.join(DATA_DIR, 'BigSolDB_filtered.csv'), index=False)
