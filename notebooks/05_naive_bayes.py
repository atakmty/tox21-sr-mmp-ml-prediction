"""
Naive Bayes for Tox21 SR-MMP
Uses PCA features from 01_data_loading_ecfp.py
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("=" * 60)
print("🧠 NAIVE BAYES (Gaussian)")
print("=" * 60)

print("\n📂 Loading preprocessed data...")
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# =====================================================================
# NAIVE BAYES (simple, no hyperparameters to tune)
# =====================================================================
print("\n🧠 Gaussian Naive Bayes with 5-fold CV...")

nb = GaussianNB()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(nb, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"✅ CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# Train & test
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_proba_nb = nb.predict_proba(X_test)[:, 1]

test_auc_nb = roc_auc_score(y_test, y_proba_nb)
print(f"\n🧪 Naive Bayes Test AUROC: {test_auc_nb:.4f}")

print("\n📊 Classification report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_nb).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp) if tp + fp > 0 else 0.0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (Naive Bayes):")
print(f"   Sensitivity (TPR): {sens:.4f}")
print(f"   Specificity (TNR): {spec:.4f}")
print(f"   Precision (PPV):   {prec:.4f}")
print(f"   F1-Score:          {f1:.4f}")

fpr_nb, tpr_nb, thr_nb = roc_curve(y_test, y_proba_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)
print(f"\n📈 ROC AUC (NB from fpr/tpr): {roc_auc_nb:.4f}")
print(f"   Thresholds: {len(thr_nb)}")

print("\n" + "=" * 60)
print("✅ NAIVE BAYES DONE")
print("=" * 60)

# AUPRC for Naive Bayes
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, y_proba_nb)
auprc_nb = auc(recall, precision)
print(f"\n📊 AUPRC (Naive Bayes): {auprc_nb:.4f}")
