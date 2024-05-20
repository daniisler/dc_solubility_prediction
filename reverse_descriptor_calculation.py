
### Determine Descriptors
"""
    Copy of descriptor calculating: just reverse data frame.
    this scripts calculates the dipole moment of all compounds within the cured data set.
    It therefor uses RDkit to calculate some conformers of each compound. The Boltzmann averaged dipole moment of the conformer ensemble is calculated

    Calculated descriptors: 
    - Dipole
    - SASA
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
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from morfeus import XTB, SASA
from morfeus.conformer import ConformerEnsemble
#from rdkit.Chem.rdmolfiles import MolToXYZFile # TODO remove
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumAromaticRings
from logger import logger
# Env
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
TMP_DIR = os.path.join(PROJECT_ROOT, 'tmp_ce_rdkit')
os.makedirs(TMP_DIR, exist_ok=True)
logger = logger.getChild('descriptor_calculation_Aq')
input_file = os.path.join(DATA_DIR, 'AqSolDB_filtered_log.csv')# TODO 
output_file = 'AqSolDB_filtered_descriptors.csv'
output_file_failed = 'AqSolDB_filtered_failed.csv'

REPLACEMENTS = {
    ord('('): 'L',
    ord(')'): 'R',
    ord('/'): '_',
    ord('\\'): 'X'
}

### Data import: _________________________________________________________________________________
df = pd.read_csv(input_file)

df = df[300:114870]# chosen randomly# TODO remove debug
df = df.iloc[::-1]# reverse data frame to run script with reversed df in parallel on euler

# Problems with qcengine and GNF-FF:
# for example the following smiles throws an error in qcengine (likely structure optimization doesn't converge)
#test_smiles = 'NS(=O)(=O)Cc1noc2ccccc12'
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
                logger.info(f"Reverse: Loading conformer ensemble for molecule with SMILES: {row['SMILES']} from cache")
                with open(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl'), 'rb') as f:
                    ce_rdkit = pickle.load(f)
                dipole_dict[row['SMILES']] = get_dipole(ce_rdkit, row["T,K"])
                logger.info(f'Calculated dipole {dipole_dict[row["SMILES"]]} for {row["SMILES"]}')
                SASA_dict[row['SMILES']] = get_SASA(ce_rdkit, row["T,K"])
                logger.info(f'Calculated SASA {SASA_dict[row["SMILES"]]} for {row["SMILES"]}')
            else:
                logger.info(f"Reverse: Calculating conformer ensemble for molecule with SMILES: {row['SMILES']}")
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
        except Exception as e:
            logger.error(f"Reverse: Error in generating conformer ensemble for molecule with SMILES: {row['SMILES']}")
            logger.error(e)
            conf_ensemble_rdkit[row['SMILES']] = 'failed'
            raise e

# Add the conformer ensemble to the data frame
df['ensemble_rdkit'] = df['SMILES'].apply(lambda x: conf_ensemble_rdkit[x] if x in conf_ensemble_rdkit.keys() else 'failed')
df['dipole'] = df['SMILES'].apply(lambda x: dipole_dict[x] if x in dipole_dict.keys() else 'failed')
df['SASA'] = df['SMILES'].apply(lambda x: SASA_dict[x] if x in SASA_dict.keys() else 'failed')
# Extract the molecules with failed conformer calculation
failed_molecules = df[(df['dipole'] == 'failed') | (df['SASA'] == 'failed')]
df = df.drop(failed_molecules.index, axis=0)

# Add the mol structure to the dataframe -> TODO: Kind of redundant, as we already have the ce?
df['mol_structure'] = df.apply(lambda x: get_mol(x['SMILES'], True), axis=1)

logger.info('Calculating rdkit descriptors...')

# Assign the list as a new column in the DataFrame
df['HBAcceptor'] = df['mol_structure'].apply(NumHAcceptors)
df['HBDonor'] = df['mol_structure'].apply(NumHDonors)
df['AromaticRings'] = df['mol_structure'].apply(NumAromaticRings)

# Save the data frame -> TODO: Most columns will be useless, as they just point to a (not existing) object
df.to_csv(os.path.join(DATA_DIR, output_file), index=False)
failed_molecules.to_csv(os.path.join(DATA_DIR, output_file_failed), index=False)
logger.info(f'Finished calculating descriptors. Data saved in {os.path.join(DATA_DIR, output_file)}')
