import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem
from pickle import load

from logger import logger

logger = logger.getChild('predict')


def predict_solubility_from_smiles(smiles, model_save_dir, T=None, solvent=None, selected_fp={'m_fp': (2048, 2)}, scale_transform=True):
    '''Predict the solubility of a molecule given its SMILES representation.

    :param str smiles: SMILES representation of the molecule
    :param str model_weights_path: path to the saved trained model
    :param float T: temperature used for filtering; None for no filtering
    :param str solvent: solvent used for filtering; None for no filtering
    :param dict of tuples selected_fp: selected fingerprint for the model, possible keys: 'm_fp', 'rd_fp', 'ap_fp', 'tt_fp'
    :param bool scale_transform: whether to scale the input data

    :return: predicted solubility (float)

    '''

    # Load the trained model
    model = torch.load(os.path.join(model_save_dir, 'architecture.pth'))
    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'weights.pth')))
    model.eval()

    # Calculate the fingerprints
    mol = MolFromSmiles(smiles)
    X = []
    for key in selected_fp.keys():
        if key == 'm_fp':
            m_fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=selected_fp[key][0], radius=selected_fp[key][1])
            X.append(torch.tensor(np.array(m_fp), dtype=torch.float32).reshape(1, -1))
        if key == 'rd_fp':
            rd_fp = Chem.RDKFingerprint(mol, fpSize=selected_fp[key][0], minPath=selected_fp[key][1][0], maxPath=selected_fp[key][1][1])
            X.append(torch.tensor(np.array(rd_fp), dtype=torch.float32).reshape(1, -1))
        if key == 'ap_fp':
            ap_fp = AllChem.GetAtomPairFingerprintAsBitVect(mol, nBits=selected_fp[key][0])
            X.append(torch.tensor(np.array(ap_fp), dtype=torch.float32).reshape(1, -1))
        if key == 'tt_fp':
            tt_fp = AllChem.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=selected_fp[key])
            X.append(torch.tensor(np.array(tt_fp), dtype=torch.float32).reshape(1, -1))
    X = np.concatenate(X, axis=1).reshape(1, -1)
    # Scale the input data according to the saved scaler
    if scale_transform:
        with open(os.path.join(model_save_dir, 'scaler.pkl'), 'rb') as f:
            scaler = load(f)
        X = scaler.transform(X)

    # Predict the solubility
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32)
        solubility = model(X).item()

    return solubility
