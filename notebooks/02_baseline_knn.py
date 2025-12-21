"""
Baseline KNN Model for Tox21 SR-MMP
Uses PCA features from 01_data_loading_ecfp.py
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("=" * 60)
print("🎯 BASELINE: K-Nearest Neighbors (KNN)")
print("=" * 60)

# Load data
print("\n📂 Loading preprocessed data...")
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# 1) Cross-validation on train set
print("\n🔄 5-fold Cross-Validation (metric: ROC-AUC)...")
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='roc_auc')  # binary ROC-AUC [web:103][web:104]

print(f"✅ CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# 2) Train on full train set and evaluate on test
print("\n🧪 Training on full train set and evaluating on test...")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]  # probability of class 1 [web:104][web:105]

test_auc = roc_auc_score(y_test, y_proba)
print(f"✅ Test AUROC: {test_auc:.4f}")

print("\n📊 Classification report:")
print(classification_report(y_test, y_pred, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp) if tp + fp > 0 else 0.0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics:")
print(f"   Sensitivity (TPR): {sensitivity:.4f}")
print(f"   Specificity (TNR): {specificity:.4f}")
print(f"   Precision (PPV):   {precision:.4f}")
print(f"   F1-Score:          {f1:.4f}")

# 3) ROC curve data (for plotting later or report)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)  # [web:122]
roc_auc = auc(fpr, tpr)
print(f"\n📈 ROC curve AUC (from fpr/tpr): {roc_auc:.4f}")
print(f"   Number of thresholds: {len(thresholds)}")

print("\n" + "=" * 60)
print("✅ KNN BASELINE DONE")
print("=" * 60)
