"""
ECFP Fingerprint Generator
Tox21 SR-MMP Project
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


def morgan_ecfp(smiles, radius=2, nbits=1024):
    """
    Generate Morgan ECFP fingerprint for SMILES string

    Parameters:
    -----------
    smiles : str
        SMILES string
    radius : int
        Radius for Morgan fingerprint (default=2, ECFP4)
    nbits : int
        Number of bits (default=1024, from outputs.txt)

    Returns:
    --------
    np.ndarray : Binary fingerprint vector
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)


def generate_ecfp_batch(smiles_list, radius=2, nbits=1024):
    """
    Generate fingerprints for multiple SMILES
    """
    fingerprints = []
    for smiles in smiles_list:
        fp = morgan_ecfp(smiles, radius, nbits)
        fingerprints.append(fp)
    return np.array(fingerprints)


def load_and_prepare_data(smiles_file, radius=2, nbits=1024):
    """
    Load SMILES file and generate fingerprints
    """
    df = pd.read_csv(smiles_file, delim_whitespace=True, header=None)
    df.columns = ['SMILES', 'ID', 'Y']

    X = generate_ecfp_batch(df['SMILES'].values, radius, nbits)
    y = df['Y'].values

    return X, y, df
