
### Detrmine Descriptors
"""
    this scripts calculates the dipole moment of all compounds within the cured data set.
    It therefor uses RDkit to calculate some conformers of each compound. The Boltzmann averaged dipole moment of the conformer ensemble is calculated

    Calculated descriptors: 
    - Dipole
    - H-acceptors/Donors
    - Aromatic Rings


    Further consideration: 
    ->  maybe could include: 
          # NHOHCount, NumHeteroatoms, and RingCount
    ->  Could determine parameters using 3D structure: 
          #SOAP? https://singroup.github.io/dscribe/latest/doc/dscribe.descriptors.html#dscribe.descriptors.soap.SOAP
          #solvent accessible surface area? https://digital-chemistry-laboratory.github.io/morfeus/sasa.html
    ->  maybe search for substructures? eg carboxylic groups.
    NOTE: could be improved by calculating conformers using CREST

"""

### Import needed packages:______________________________________________________________________
import os
import sys
from logger import logger
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from morfeus import XTB
from morfeus.conformer import ConformerEnsemble
from rdkit.Chem.rdmolfiles import MolToXYZFile
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumAromaticRings

# Env
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
logger = logger.getChild('descriptor_calculation')
input_file = os.path.join(DATA_DIR, 'BigSolDB_filtered.csv')

### Data import: _________________________________________________________________________________
df = pd.read_csv(input_file)

# for example the following smiles throws an error in qcengine (likely structure optimization doesn't converge)
test_smiles = 'NS(=O)(=O)Cc1noc2ccccc12'

# Display the problematic molecule
# molecule = Chem.MolFromSmiles(excluded_smiles)
# img = Draw.MolToImage(molecule)
# img.show()

# TODO: Remove me, only for testing!
# df = df[:5]

### Functions:__________________________________________________________________________________________________
def ce_from_rdkit(smiles):
    """
    This functions generates random conformers based on the smiles
    and gets rid of redundant conformers based on RMSD. The remaining conformers
    are optimised using xtb on a level of GFN-FF (a force-field). The energy
    of the optimised geometry is determined with xtb on a level of GFN2-xTB.
    The energetically most favorable conformers are then selected.


    Adapted from ex. 4
    """
    # Generate MORFEUS Conformer Ensemble from RDKit and prune by RMSD
    # MMFF94s reflects the time averaged structure better, which is what we need
    ce_rdkit = ConformerEnsemble.from_rdkit(smiles, n_confs=10, optimize="MMFF94s")
    ce_rdkit.prune_rmsd()

    # Optimise all of the remaining conformers and sort them energetically
    model={"method": "GFN1-xTB"}
    ce_rdkit.optimize_qc_engine(program="xtb", model=model, procedure="berny")
    ce_rdkit.sort()

    # Single point energy calculation and final energetic sorting
    model={"method": "GFN2-xTB"}
    ce_rdkit.sp_qc_engine(program="xtb", model=model)
    ce_rdkit.sort()
    # Generate molecule representation
    #ce_rdkit.generate_mol()
    logger.info(f"Conformer ensemble for molecule with SMILES: {smiles} generated successfully")
    return ce_rdkit

# Function to generate bolzman average dipole from conformer ensemble
def get_dipole(ce):
    for conformer in ce:
        xtb = XTB(conformer.elements, conformer.coordinates)
        dipole = xtb.get_dipole() # gives array as 3D vector
        conformer.properties["dipole"] = np.linalg.norm(dipole)

    return ce.boltzmann_statistic("dipole")

# Get molecular structure using rdkit
def get_mol(smiles, get_Hs = True):
    mol = Chem.MolFromSmiles(smiles)
    mol_hasHs = Chem.AddHs(mol)

    if get_Hs:
        return(mol_hasHs)
    else:
        return(mol)

### Calculate Descriptors ______________________________________________________________________________
conf_ensemble_rdkit = {}
# Calculate conformer ensemble for all molecules to obtain the dipole moments
for index, row in df.iterrows():
    if row['SMILES'] not in conf_ensemble_rdkit.keys():
        logger.info(f"Calculating conformer ensemble for molecule with SMILES: {row['SMILES']}")
        try:
            ce_rdkit = ce_from_rdkit(row['SMILES'])
            conf_ensemble_rdkit[row['SMILES']] = ce_rdkit
        except Exception as e:
            logger.error(f"Error in generating conformer ensemble for molecule with SMILES: {row['SMILES']}")
            logger.error(e)
            conf_ensemble_rdkit[row['SMILES']] = 'failed'

# Add the conformer ensemble to the dataframe
df['ensemble_rdkit'] = df['SMILES'].apply(lambda x: conf_ensemble_rdkit[x] if x in conf_ensemble_rdkit.keys() else 'failed')
# Extract the molecules with failed conformer calculation
failed_molecules = df[df['ensemble_rdkit'] == 'failed']
df = df.drop(failed_molecules.index, axis=0)

logger.info('Calculating dipole moment for all successfully generated conformers')

df['dipole'] = df['ensemble_rdkit'].apply(get_dipole)

# Add the mol structure to the dataframe -> TODO: Kind of redundant, as we already have the ce?
df['mol_structure'] = df.apply(lambda x: get_mol(x['SMILES'], True), axis=1)

logger.info('Calculating rdkit descriptors...')

# Assign the list as a new column in the DataFrame
df['HBAcceptor'] = df['mol_structure'].apply(NumHAcceptors)
df['HBDonor'] = df['mol_structure'].apply(NumHDonors)
df['AromaticRings'] = df['mol_structure'].apply(NumAromaticRings)

# Save the dataframe -> TODO: Most columns will be useless, as they just point to a (not existing) object
output_file = 'BigSolDB_filtered_descriptors.csv'
df.to_csv(os.path.join(DATA_DIR, output_file), index=False)
failed_molecules.to_csv(os.path.join(DATA_DIR, 'BigSolDB_filtered_failed.csv'), index=False)
logger.info(f'Finished calculating descriptors. Data saved in {os.path.join(DATA_DIR, output_file)}')
