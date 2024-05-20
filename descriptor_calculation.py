### Determine Descriptors
"""
    This scripts calculates the the temperature dependent, boltzmann average of the dipole moment & the SASA using rdkit and morfeus.
    
    Calculated descriptors:
    - Dipole
    - SASA (SolventAccessibleSurfaceArea)
    Optional: H-bond Donor/Acceptor and aromatic ring count
"""

### Import needed packages:______________________________________________________________________
import os
import pickle
import sys

from logger import logger

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from morfeus import XTB, SASA
from morfeus.conformer import ConformerEnsemble
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumAromaticRings

# Env
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
TMP_DIR = os.path.join(PROJECT_ROOT, 'tmp_ce_rdkit')
os.makedirs(TMP_DIR, exist_ok=True)
logger = logger.getChild('descriptor_calculation_Aq')
input_file = os.path.join(DATA_DIR, 'AqSolDB_filtered_log.csv')# TODO
output_file = 'AqSolDB_filtered_descriptors.csv'
output_file_failed = 'AqSolDB_filtered_failed.csv'
# add_Lipinski_descriptors: If True, calculate H-bond donor/acceptor groups and aromatic ring count and add to data frame
add_Lipinski_descriptors = False
# add_mol_structure: If True, add mol structure to data frame
add_mol_structure = False
# ce_calculation: If True, calculate conformer ensemble when no respective "*_ce_rdkit.pkl" is available. 
ce_calculation = False

REPLACEMENTS = {
    ord('('): 'L',
    ord(')'): 'R',
    ord('/'): '_',
    ord('\\'): 'X'
}

### Data import: _________________________________________________________________________________
df = pd.read_csv(input_file)

###TODO entfernen: 
# df = df[df['weight'] > 0.3] # Filter df for accuracy
# # cut df
# df = df[300:]# chosen randomly

# Problems with qcengine and GNF-FF:
# for example the following smiles throws an error in qcengine (likely structure optimization doesn't converge)
#test_smiles = ['NS(=O)(=O)Cc1noc2ccccc12', C#CC#CC=C=CCCO]
# Display the problematic molecule:
# molecule = Chem.MolFromSmiles(excluded_smiles)
# img = Draw.MolToImage(molecule)
# img.show()

### Functions:__________________________________________________________________________________________________

# # Parameters passed to the conformer ensemble calculation (new local_options which is deprecated)
# class TaskConfig(pydantic.BaseSettings):
#     """Description of the configuration used to launch a task."""

#     # Specifications
#     ncores: int = pydantic.Field(None, description="Number cores per task on each node")
#     nnodes: int = pydantic.Field(None, description="Number of nodes per task")
#     memory: float = pydantic.Field(
#         None, description="Amount of memory in GiB (2^30 bytes; not GB = 10^9 bytes) per node."
#     )
#     scratch_directory: Optional[str]  # What location to use as scratch
#     retries: int  # Number of retries on random failures
#     mpiexec_command: Optional[str]  # Command used to launch MPI tasks, see NodeDescriptor
#     use_mpiexec: bool = False  # Whether it is necessary to use MPI to run an executable
#     cores_per_rank: int = pydantic.Field(1, description="Number of cores per MPI rank")
#     scratch_messy: bool = pydantic.Field(
#         False, description="Leave scratch directory and contents on disk after completion."
#     )

#     class Config(pydantic.BaseSettings.Config):
#         extra = "forbid"
#         env_prefix = "QCENGINE_"


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
    f_ce_rdkit = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94s")
    f_ce_rdkit.prune_rmsd()

    # Optimise all of the remaining conformers and sort them energetically
    model = {"method": "GFN1-xTB"}
    f_ce_rdkit.optimize_qc_engine(program="xtb", model=model, procedure="berny", local_options={"ncores": 1, "nnodes": 1, "cores_per_rank": 1})
    f_ce_rdkit.sort()

    # Single point energy calculation and final energetic sorting
    model = {"method": "GFN2-xTB"}
    f_ce_rdkit.sp_qc_engine(program="xtb", model=model)
    f_ce_rdkit.sort()
    return f_ce_rdkit


# Function to generate Boltzmann averaged dipole from conformer ensemble
def get_dipole(ce, temp):
    for conformer in ce:
        xtb = XTB(conformer.elements, conformer.coordinates)
        dipole = xtb.get_dipole() # gives array as 3D vector
        conformer.properties["dipole"] = np.linalg.norm(dipole)
    return ce.boltzmann_statistic("dipole", temperature=temp, statistic='avg')


# Function to get Boltzmann averaged SolventAccessibleSurfaceArea (SASA) from conformer ensemble: https://digital-chemistry-laboratory.github.io/morfeus/conformer.html
def get_SASA(ce, temp):
    for conformer in ce:
        sasa = SASA(ce.elements, conformer.coordinates)
        conformer.properties["sasa"] = sasa.area
        ce.boltzmann_weights()
    return ce.boltzmann_statistic("sasa", temperature=temp, statistic='avg')


# Get molecular structure using rdkit
def get_mol(smiles, get_Hs = True):
    mol = Chem.MolFromSmiles(smiles)
    mol_hasHs = Chem.AddHs(mol)

    if get_Hs:
        return mol_hasHs
    return mol


# Make smiles to filename:
def smiles_to_file(smiles):
    # replace: (see REPLACEMENTS defined above)
    # (     --> L
    # )     --> R
    # /     --> _
    # \\    --> X
    return smiles.translate(REPLACEMENTS)


### Calculate Descriptors ______________________________________________________________________________
conf_ensemble_rdkit = {}
dipole_dict = {}
SASA_dict = {}

# Calculate conformer ensemble for all molecules to obtain the dipole moments
for index, row in df.iterrows():
    if row['SMILES'] not in conf_ensemble_rdkit.keys():
        try:
            # Check if the conformer ensemble has already been calculated
            smiles_identifier = smiles_to_file(row["SMILES"])
            if os.path.exists(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl')):
                logger.info(f"Loading conformer ensemble for molecule with SMILES: {row['SMILES']} from cache")
                with open(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl'), 'rb') as f:
                    ce_rdkit = pickle.load(f)
                conf_ensemble_rdkit[row['SMILES']] = ce_rdkit
                dipole_dict[row['SMILES']] = get_dipole(ce_rdkit, row["T,K"])
                logger.info(f'Calculated dipole {dipole_dict[row["SMILES"]]} for {row["SMILES"]}')
                SASA_dict[row['SMILES']] = get_SASA(ce_rdkit, row["T,K"])
                logger.info(f'Calculated SASA {SASA_dict[row["SMILES"]]} for {row["SMILES"]}')
            else:
                if ce_calculation:
                    logger.info(f"Calculating conformer ensemble for molecule with SMILES: {row['SMILES']}")
                    ce_rdkit = ce_from_rdkit(row['SMILES'])
                    conf_ensemble_rdkit[row['SMILES']] = ce_rdkit
                    if not ce_rdkit == 'failed':
                        # Save the conformer ensemble to a file
                        with open(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl'), 'wb') as f:
                            pickle.dump(ce_rdkit, f)
                        dipole_dict[row['SMILES']] = get_dipole(ce_rdkit, row["T,K"])
                        logger.info(f'Calculated dipole {dipole_dict[row["SMILES"]]} for {row["SMILES"]}')
                        SASA_dict[row['SMILES']] = get_SASA(ce_rdkit, row["T,K"])
                        logger.info(f'Calculated SASA {SASA_dict[row["SMILES"]]} for {row["SMILES"]}')
                else: 
                    logger.info(f"No conformer ensemble for molecule with SMILES: {row['SMILES']}available. No Descriptor calculation performed.")
                    conf_ensemble_rdkit[row['SMILES']] = 'failed'

        except Exception as e:
            logger.error(f"Error in generating conformer ensemble for molecule with SMILES: {row['SMILES']}")
            logger.error(e)
            conf_ensemble_rdkit[row['SMILES']] = 'failed'
            #raise e
            continue

# Add the conformer ensemble to the data frame
df['ensemble_rdkit'] = df['SMILES'].apply(lambda x: conf_ensemble_rdkit[x] if x in conf_ensemble_rdkit.keys() else 'failed')
df['dipole'] = df['SMILES'].apply(lambda x: dipole_dict[x] if x in dipole_dict.keys() else 'failed')
df['SASA'] = df['SMILES'].apply(lambda x: SASA_dict[x] if x in SASA_dict.keys() else 'failed')

# Extract the molecules with failed conformer calculation
failed_molecules = df[(df['dipole'] == 'failed') | (df['SASA'] == 'failed')]
df = df.drop(failed_molecules.index, axis=0)

if add_mol_structure:
    # Add the mol structure to the data frame
    df['mol_structure'] = df.apply(lambda x: get_mol(x['SMILES'], True), axis=1)

# Assign the list as a new column in the DataFrame
if add_Lipinski_descriptors:
    logger.info('Calculating rdkit descriptors...')
    df['HBAcceptor'] = df['mol_structure'].apply(NumHAcceptors)
    df['HBDonor'] = df['mol_structure'].apply(NumHDonors)
    df['AromaticRings'] = df['mol_structure'].apply(NumAromaticRings)

# Save the data frame
df.to_csv(os.path.join(DATA_DIR, output_file), index=False)
failed_molecules.to_csv(os.path.join(DATA_DIR, output_file_failed), index=False)
logger.info(f'Finished calculating descriptors. Data saved in {os.path.join(DATA_DIR, output_file)}')
