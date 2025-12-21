"""
Data Loading & ECFP Fingerprint Generation
Tox21 SR-MMP Toxicity Prediction
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import re
import os

print("="*60)
print("📊 TOX21 SR-MMP: Data Loading & ECFP Feature Extraction")
print("="*60)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n🔍 STEP 1: Loading SR-MMP.smiles...")

def parse_smiles_line(line):
    """
    Parse SMILES line: SMILES (start) + ID (middle) + Y (end)
    """
    parts = re.split(r'\s+', line.strip())
    smiles = parts[0]  # First: SMILES
    y = int(parts[-1]) # Last: Label (0/1)
    return smiles, y

# Read file
smiles_list, y_list = [], []
with open('SR-MMP.smiles', 'r') as f:
    for line in f:
        smiles, y = parse_smiles_line(line)
        smiles_list.append(smiles)
        y_list.append(y)

df = pd.DataFrame({'Drug': smiles_list, 'Y': y_list})
print(f"✅ Shape: {df.shape}")
print(f"☠️ Toxic compounds: {(df['Y']==1).sum()} ({(df['Y']==1).mean():.1%})")
print(f"\n📋 First 3 rows:")
print(df.head(3))

# ============================================================================
# STEP 2: GENERATE MORGAN ECFP (Proposal Section 3.1)
# ============================================================================
print("\n🔬 STEP 2: Generating Morgan ECFP fingerprints...")
print("   Parameters: radius=2, nBits=1024 (from outputs.txt)")

def morgan_ecfp(smiles, radius=2, nBits=1024):
    """
    Generate Morgan ECFP fingerprint for single SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

# Generate for all compounds
print("   Computing fingerprints... (~3-5 minutes)")
df['ECFP'] = df['Drug'].apply(morgan_ecfp)
X = np.stack(df['ECFP'].values)

print(f"✅ ECFP Complete!")
print(f"   X shape: {X.shape}")
print(f"   Memory: {X.nbytes / 1e6:.1f} MB")
print(f"   Sample fingerprint (first 10 bits): {X[0][:10]}")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT (Proposal Section 3.2)
# ============================================================================
print("\n✂️  STEP 3: Train/Test split (80/20, stratified)...")

y = df['Y'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"✅ Split Complete!")
print(f"   Train: {X_train.shape} ({y_train.mean():.1%} toxic)")
print(f"   Test:  {X_test.shape}  ({y_test.mean():.1%} toxic)")
print(f"   ✓ Stratified: toxic ratios match!")

# ============================================================================
# STEP 4: PCA DIMENSIONALITY REDUCTION (Proposal Section 4)
# ============================================================================
print("\n📉 STEP 4: PCA dimensionality reduction...")
print("   n_components=100 (from outputs.txt)")

pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"✅ PCA Complete!")
print(f"   Train shape: {X_train_pca.shape}")
print(f"   Test shape:  {X_test_pca.shape}")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
print(f"   Top 5 components: {pca.explained_variance_ratio_[:5].sum():.1%}")

# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================
print("\n💾 STEP 5: Saving preprocessed data...")

np.save('X_train_pca.npy', X_train_pca)
np.save('X_test_pca.npy', X_test_pca)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"✅ Saved:")
print(f"   - X_train_pca.npy")
print(f"   - X_test_pca.npy")
print(f"   - y_train.npy")
print(f"   - y_test.npy")

print("\n" + "="*60)
print("✅ DATA PREPROCESSING COMPLETE!")
print("   Ready for model training (KNN, SVM, etc.)")
print("="*60)
