
### Detrmine Descriptors
"""
    this scripts calculates the dipole moment of all compounds within the cured data set.
    It therefor uses RDkit to calculate some conformers of each compound. The boltzmann averaged dipolemoment of the conformer ensemble is calculated

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

### Import needed packages: 
import os

#### The following code copied from Google colab. Not compatibel with py script in VisStudioCode??
'''
# Import of needed functions in the terminal:
%%bash

# Install some python packages that we use throughout the exercise
pip install rdkit
pip install morfeus-ml
pip install pyberny
pip install qcengine
pip install xtb

# Could improve by using crest...
#Download miniforge, our way of installing conda
#wget -c https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#chmod +x Miniforge3-Linux-x86_64.sh
#bash ./Miniforge3-Linux-x86_64.sh -b -f -p /usr/local

# Use conda to install CREST
#conda install -q -y -c conda-forge crest
'''
'''
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZFile
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
import numpy as np
# to calculate some describtors: https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html#rdkit.Chem.Lipinski.NumHAcceptors
from rdkit.Chem.Lipinski import NumHAcceptors
from rdkit.Chem.Lipinski import NumHDonors
from rdkit.Chem.Lipinski import NumAromaticRings
'''

# importet via terminal..?? python -c "from rdkit import Chem"
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZFile
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
from morfeus import XTB
"""

### Data import: # not sure if this works...
PROJECT_ROOT = os.path.dirname(os.path.abspath("input_data/.combined_cleaned_data.csv.icloud"))
df = os.path.join(PROJECT_ROOT,'cure')


### Functions:____________________________________________________________________________________________________________________
def ce_from_rdkit(smiles):
    """
    This functions generates random conformers beased on the smiles
    and gets rid of redunant conformers based on RMSD. The remaining conformers
    are optimised using xtb on a level of GFN-FF (a force-field). The energy
    of the optimised geometry is determined with xtb on a level of GFN2-xTB.
    The energetically most favorable conformers are then selected.


    Mostly took from ex. 4
    """
    # Generate MORFEUS Conformer Ensemble from RDKit and prune by RMSD
    ce_rdkit = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
    ce_rdkit.prune_rmsd()

    # Optimise all of the remaining conformers and sort them energetically
    model={"method": "GFN-FF"}
    ce_rdkit.optimize_qc_engine(program="xtb", model=model, procedure="berny")
    ce_rdkit.sort()

    # Single point energy calculation and final energetic sorting
    model={"method": "GFN2-xTB"}
    ce_rdkit.sp_qc_engine(program="xtb", model=model)
    ce_rdkit.sort()
    # Generate molecule representation
    #ce_rdkit.generate_mol()

    return ce_rdkit


# Function to generate bolzman average dipole from conformer ensemble
def get_dipole(ce):
  for conformer in ce:
    xtb = XTB(conformer.elements, conformer.coordinates)
    dipole = xtb.get_dipole()# gives array as 3D vector
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

# Get some further descriptors from rdkit
def get_HAcceptors(mol): 
  HBA = NumHAcceptors(mol)#CalcNumLipinskiHBA(mol)
  return(HBA)

def get_HDonors(mol): 
  HBD = NumHDonors(mol)
  return(HBD)

def get_AromaticRings(mol):
  return(NumAromaticRings(mol))


### Get Descriptors ______________________________________________________________________________________________________________
# Initialisations:
conf_ensemble_rdkit = []
mol_rdkit = []
HBA_rdkit = []
HBD_rdkit = []
AromatRi_rdkit = []


##...Add Conformer ensemble:
for index, row in df.iterrows():# itterate over rows

    # Get the ConformerEnsemble from ce_from_crest
    rdkit_value = ce_from_rdkit(row['smiles'])
    # Append the value to the list
    conf_ensemble_rdkit.append(rdkit_value)

# Assign the list as a new column in the acids DataFrame
df['ensemble_rdkit'] = conf_ensemble_rdkit

##...get dipole from conformer in new column: 
df['dipole'] = df['ensemble_crest'].apply(get_dipole)


##...Add Molecular Structure:
for index, row in df.iterrows():

    # Get molecular structure
    rdkit_mol = get_mol(row['smiles'], True)
    # Append mol to the list
    mol_rdkit.append(rdkit_mol)

# Assign the list as a new column in the DataFrame
df['mol_structure'] = mol_rdkit


##...Add rdkit Descriptors:
for index, row in df.iterrows():

    # Get the Descriptors for each row
    rdkit_HBA = get_HAcceptors(row['mol_structure'])
    rdkit_HBD = get_HAcceptors(row['mol_structure'])
    rdkit_AromatRi = get_AromaticRings(row['mol_structure'])

    # Append the value to the list
    HBA_rdkit.append(rdkit_HBA)
    HBD_rdkit.append(rdkit_HBD)
    AromatRi_rdkit.append(rdkit_AromatRi)

# Assign the list as a new column in the DataFrame
df['HBAcceptor'] = HBA_rdkit
df['HBDonor'] = HBD_rdkit
df['AromaticRings'] = AromatRi_rdkit