import os
import sys
from logger import logger
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from morfeus import XTB
from morfeus.conformer import ConformerEnsemble
import pickle
from rdkit.Chem.rdmolfiles import MolToXYZFile
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumAromaticRings

# Env
PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'input_data')
TMP_DIR = os.path.join(PROJECT_ROOT, 'tmp_ce_rdkit')
os.makedirs(TMP_DIR, exist_ok=True)
logger = logger.getChild('descriptor_calculation_Aq')

smiles_test= 'Br/C(C=NNc1ccc2nncn2n1)=C\c1ccccc1'

REPLACEMENTS = {
    ord('('): 'L',
    ord(')'): 'R',
    ord('/'): '_',
    ord('\\'): 'X'
}

### Data import: _________________________________________________________________________________


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
    ce_rdkit = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94s")
    ce_rdkit.prune_rmsd()

    # Optimise all of the remaining conformers and sort them energetically
    model={"method": "GFN2-xTB"}
    ce_rdkit.optimize_qc_engine(program="xtb", model=model, procedure="berny", local_options={"ncores": 1, "nnodes": 1, "cores_per_rank": 1})# TODO: need to de-comment above defined Class I think
    #sys.exit()#TODO debugging? to be removed
    ce_rdkit.sort()

    # Single point energy calculation and final energetic sorting
    model={"method": "GFN2-xTB"}
    ce_rdkit.sp_qc_engine(program="xtb", model=model)
    ce_rdkit.sort()
    return ce_rdkit

# Function to generate boltzman average dipole from conformer ensemble
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
# Calculate conformer ensemble for all molecules to obtain the dipole moments

if smiles_test not in conf_ensemble_rdkit.keys():
    try:
        logger.info(f"Calculating conformer ensemble for molecule with SMILES:{smiles_test}")
        ce_rdkit = ce_from_rdkit(smiles_test)
        
        # Check if the conformer ensemble has already been calculated
        # smiles_identifier = smiles_to_file(smiles_test)
        # if os.path.exists(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl')):
        #     logger.info(f"Loading conformer ensemble for molecule with SMILES: {row['SMILES']} from cache")
        #     with open(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl'), 'rb') as f:
        #         ce_rdkit = pickle.load(f)
        #     dipole_dict[row['SMILES']] = get_dipole(ce_rdkit)
        #     logger.info(f'Calculated dipole {dipole_dict[smiles_test]} for {smiles_test}')
        # else:
        #     logger.info(f"Calculating conformer ensemble for molecule with SMILES: {row['SMILES']}")
        #     ce_rdkit = ce_from_rdkit(row['SMILES'])
        #     conf_ensemble_rdkit[row['SMILES']] = ce_rdkit
        #     if not ce_rdkit == 'failed':
        #         # Save the conformer ensemble to a file
        #         with open(os.path.join(TMP_DIR, f'{smiles_identifier}_ce_rdkit.pkl'), 'wb') as f:
        #             pickle.dump(ce_rdkit, f)
        #         dipole_dict[row['SMILES']] = get_dipole(ce_rdkit)
        #         logger.info(f'Calculated dipole {dipole_dict[smiles_test]} for {smiles_test}')
    except Exception as e:
        logger.error(f"Error in generating conformer ensemble for molecule with SMILES: {smiles_test}")
        logger.error(e)
        conf_ensemble_rdkit[smiles_test] = 'failed'
        raise e

