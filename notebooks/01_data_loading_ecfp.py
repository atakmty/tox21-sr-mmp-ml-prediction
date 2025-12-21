"""
Data Loading & ECFP Fingerprint Generation (Separate Train/Test Files)
Tox21 SR-MMP Toxicity Prediction
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import os

print("="*70)
print("📊 TOX21 SR-MMP: Separate Train/Test Data Loading & ECFP Generation")
print("="*70)

# ============================================================================
# STEP 1: LOAD TRAIN & TEST DATA (TSV Format)
# ============================================================================
print("\n🔍 STEP 1: Loading sr-mmp-train.smiles & sr-mmp-test.smiles...")

def load_smiles_file(filepath):
    """Load TSV SMILES file (SMILES, ID, Label)"""
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['SMILES', 'CompoundID', 'Label']
    )
    return df

# Load train & test
df_train = load_smiles_file('data/sr-mmp-train.smiles')
df_test = load_smiles_file('data/sr-mmp-test.smiles')

print(f"✅ Train set: {df_train.shape}")
print(f"   ☠️ Toxic: {(df_train['Label']==1).sum()} ({(df_train['Label']==1).mean():.1%})")

print(f"\n✅ Test set: {df_test.shape}")
print(f"   ☠️ Toxic: {(df_test['Label']==1).sum()} ({(df_test['Label']==1).mean():.1%})")

print(f"\n📋 Train set - First 2 rows:")
print(df_train.head(2))
print(f"\n📋 Test set - First 2 rows:")
print(df_test.head(2))

# ============================================================================
# STEP 2: GENERATE MORGAN ECFP (Proposal Section 3.1)
# ============================================================================
print("\n🔬 STEP 2: Generating Morgan ECFP fingerprints...")
print("   Parameters: radius=2 (ECFP4), nBits=1024")

def morgan_ecfp(smiles, radius=2, nbits=1024):
    """
    Generate Morgan ECFP fingerprint for single SMILES
    Returns zero vector if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)

# Generate ECFP for train set
print("\n   Computing ECFP for TRAIN set (~2,073 compounds)...")
df_train['ECFP'] = df_train['SMILES'].apply(morgan_ecfp)
X_train = np.stack(df_train['ECFP'].values)

print(f"   ✅ X_train: {X_train.shape}")
print(f"      Memory: {X_train.nbytes / 1e6:.1f} MB")
print(f"      Sample: {X_train[0][:10]}")

# Generate ECFP for test set
print("\n   Computing ECFP for TEST set (~78 compounds)...")
df_test['ECFP'] = df_test['SMILES'].apply(morgan_ecfp)
X_test = np.stack(df_test['ECFP'].values)

print(f"   ✅ X_test: {X_test.shape}")
print(f"      Memory: {X_test.nbytes / 1e6:.1f} MB")

y_train = df_train['Label'].values
y_test = df_test['Label'].values

# ============================================================================
# STEP 3: PCA DIMENSIONALITY REDUCTION (Proposal Section 4)
# ============================================================================
print("\n📉 STEP 3: PCA dimensionality reduction...")
print("   n_components=100")

pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"✅ PCA Complete!")
print(f"   Train shape: {X_train_pca.shape}")
print(f"   Test shape:  {X_test_pca.shape}")
print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
print(f"   Top 5 components: {pca.explained_variance_ratio_[:5].sum():.1%}")

# ============================================================================
# STEP 4: DATA SUMMARY
# ============================================================================
print("\n📊 STEP 4: Data Summary")
print("="*70)
print(f"{'Train Set':<20} {X_train_pca.shape[0]:>10} samples | Toxic: {y_train.mean():>6.1%}")
print(f"{'Test Set':<20} {X_test_pca.shape[0]:>10} samples | Toxic: {y_test.mean():>6.1%}")
print(f"{'Features (after PCA)':<20} {X_train_pca.shape[1]:>10} dimensions")
print("="*70)

# ============================================================================
# STEP 5: SAVE PREPROCESSED DATA
# ============================================================================
print("\n💾 STEP 5: Saving preprocessed data...")

np.save('X_train_pca.npy', X_train_pca)
np.save('X_test_pca.npy', X_test_pca)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save metadata
metadata = {
    'n_train_samples': len(y_train),
    'n_test_samples': len(y_test),
    'n_features_original': X_train.shape[1],
    'n_features_pca': X_train_pca.shape[1],
    'pca_variance': pca.explained_variance_ratio_.sum(),
    'train_toxic_rate': y_train.mean(),
    'test_toxic_rate': y_test.mean()
}

import json
with open('data_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved:")
print(f"   - X_train_pca.npy ({X_train_pca.nbytes / 1e6:.1f} MB)")
print(f"   - X_test_pca.npy ({X_test_pca.nbytes / 1e6:.1f} MB)")
print(f"   - y_train.npy")
print(f"   - y_test.npy")
print(f"   - data_metadata.json")

print("\n" + "="*70)
print("✅ DATA PREPROCESSING COMPLETE!")
print("   Ready for model training (KNN, SVM, Ensemble, etc.)")
print("="*70)
