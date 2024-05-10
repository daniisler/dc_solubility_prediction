import os
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem
from pickle import load

from nn_model import SolubilityModel
from logger import logger

logger = logger.getChild('predict')


def predict_solubility_from_smiles(smiles, model_save_dir, best_hyperparams, T=None, solvents=None, selected_fp={'m_fp': (2048, 2)}, scale_transform=True):
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
    # model = torch.load(os.path.join(model_save_dir, 'architecture.pth'))
    model = SolubilityModel(
        train_data=None,
        valid_data=None,
        test_data=None,
        input_size=sum([selected_fp[key][0] for key in selected_fp.keys()]),
        n_neurons_hidden_layers=best_hyperparams['n_neurons_hidden_layers'],
        activation_function=best_hyperparams['activation_fn'],
        loss_function=best_hyperparams['loss_fn'],
        optimizer=best_hyperparams['optimizer'],
        lr=best_hyperparams['learning_rate'],
        batch_size=best_hyperparams['batch_size'],
    )

    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'weights.pth')))
    model.eval()

    # Calculate the fingerprints
    mol = MolFromSmiles(smiles)
    if solvents:
        mols_solvents = [Chem.MolFromSmiles(solvent) for solvent in solvents]
    X = []
    for key in selected_fp.keys():
        if key == 'm_fp':
            m_fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=selected_fp[key][0], radius=selected_fp[key][1])
            X.append(torch.tensor(np.array(m_fp), dtype=torch.float32).reshape(1, -1))
            if solvents:
                m_fp_solvents = [AllChem.GetMorganFingerprintAsBitVect(mol_s, nBits=selected_fp[key][0], radius=selected_fp[key][1]) for mol_s in mols_solvents]
                for m_fp_s in m_fp_solvents:
                    X.append(torch.tensor(np.array(m_fp_s), dtype=torch.float32).reshape(1, -1))
        if key == 'rd_fp':
            rd_fp = Chem.RDKFingerprint(mol, fpSize=selected_fp[key][0], minPath=selected_fp[key][1][0], maxPath=selected_fp[key][1][1])
            X.append(torch.tensor(np.array(rd_fp), dtype=torch.float32).reshape(1, -1))
            if solvents:
                rd_fp_solvents = [Chem.RDKFingerprint(mol_s, fpSize=selected_fp[key][0], minPath=selected_fp[key][1][0], maxPath=selected_fp[key][1][1]) for mol_s in mols_solvents]
                for rd_fp_s in rd_fp_solvents:
                    X.append(torch.tensor(np.array(rd_fp_s), dtype=torch.float32).reshape(1, -1))
        if key == 'ap_fp':
            ap_fp = AllChem.GetAtomPairFingerprintAsBitVect(mol, nBits=selected_fp[key][0])
            X.append(torch.tensor(np.array(ap_fp), dtype=torch.float32).reshape(1, -1))
            if solvents:
                ap_fp_solvents = [AllChem.GetAtomPairFingerprintAsBitVect(mol_s, nBits=selected_fp[key][0]) for mol_s in mols_solvents]
                for ap_fp_s in ap_fp_solvents:
                    X.append(torch.tensor(np.array(ap_fp_s), dtype=torch.float32).reshape(1, -1))
        if key == 'tt_fp':
            tt_fp = AllChem.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=selected_fp[key])
            X.append(torch.tensor(np.array(tt_fp), dtype=torch.float32).reshape(1, -1))
            if solvents:
                tt_fp_solvents = [AllChem.GetTopologicalTorsionFingerprintAsBitVect(mol_s, nBits=selected_fp[key]) for mol_s in mols_solvents]
                for tt_fp_s in tt_fp_solvents:
                    X.append(torch.tensor(np.array(tt_fp_s), dtype=torch.float32).reshape(1, -1))
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
