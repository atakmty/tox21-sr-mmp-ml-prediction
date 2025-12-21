"""
SVM Models for Tox21 SR-MMP
Uses PCA features from 01_data_loading_ecfp.py
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("=" * 60)
print("🧪 SVM MODELS: RBF & Linear")
print("=" * 60)

print("\n📂 Loading preprocessed data...")
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# ---------------------------------------------------------------------
# 1) RBF SVM + hiperparametre araması
# ---------------------------------------------------------------------
print("\n🔎 RBF SVM: hyperparameter search (C, gamma) with 5-fold CV...")

param_grid_rbf = {
    'C':    [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}

rbf_svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_rbf = GridSearchCV(
    rbf_svm,
    param_grid_rbf,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)

grid_rbf.fit(X_train, y_train)
print(f"✅ Best params (RBF): {grid_rbf.best_params_}")
print(f"✅ Best CV AUROC (RBF): {grid_rbf.best_score_:.4f}")

# Test performansı
best_rbf = grid_rbf.best_estimator_
y_pred_rbf = best_rbf.predict(X_test)
y_proba_rbf = best_rbf.predict_proba(X_test)[:, 1]

test_auc_rbf = roc_auc_score(y_test, y_proba_rbf)
print(f"\n🧪 RBF SVM Test AUROC: {test_auc_rbf:.4f}")

print("\n📊 Classification report (RBF SVM):")
print(classification_report(y_test, y_pred_rbf, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rbf).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp) if tp + fp > 0 else 0.0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (RBF SVM):")
print(f"   Sensitivity (TPR): {sens:.4f}")
print(f"   Specificity (TNR): {spec:.4f}")
print(f"   Precision (PPV):   {prec:.4f}")
print(f"   F1-Score:          {f1:.4f}")

fpr_rbf, tpr_rbf, thr_rbf = roc_curve(y_test, y_proba_rbf)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)
print(f"\n📈 ROC AUC (RBF from fpr/tpr): {roc_auc_rbf:.4f}")
print(f"   Thresholds: {len(thr_rbf)}")

# AUPRC for RBF SVM
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, y_proba_rbf)
auprc_rbf = auc(recall, precision)
print(f"\n📊 AUPRC (RBF SVM): {auprc_rbf:.4f}")

# ---------------------------------------------------------------------
# 2) Linear SVM (daha hızlı, baseline)
# ---------------------------------------------------------------------
print("\n⚡ Linear SVM (SVC(kernel='linear'))...")

lin_svm = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
lin_svm.fit(X_train, y_train)

y_pred_lin = lin_svm.predict(X_test)
y_proba_lin = lin_svm.predict_proba(X_test)[:, 1]

test_auc_lin = roc_auc_score(y_test, y_proba_lin)
print(f"✅ Linear SVM Test AUROC: {test_auc_lin:.4f}")

print("\n📊 Classification report (Linear SVM):")
print(classification_report(y_test, y_pred_lin, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lin).ravel()
sens_l = tp / (tp + fn)
spec_l = tn / (tn + fp)
prec_l = tp / (tp + fp) if tp + fp > 0 else 0.0
f1_l = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (Linear SVM):")
print(f"   Sensitivity (TPR): {sens_l:.4f}")
print(f"   Specificity (TNR): {spec_l:.4f}")
print(f"   Precision (PPV):   {prec_l:.4f}")
print(f"   F1-Score:          {f1_l:.4f}")

fpr_lin, tpr_lin, thr_lin = roc_curve(y_test, y_proba_lin)
roc_auc_lin = auc(fpr_lin, tpr_lin)
print(f"\n📈 ROC AUC (Linear from fpr/tpr): {roc_auc_lin:.4f}")
print(f"   Thresholds: {len(thr_lin)}")

print("\n" + "=" * 60)
print("✅ SVM MODELS DONE")
print("=" * 60)

# AUPRC for Linear SVM (sonuna ekle)
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, y_proba_lin)
auprc_lin = auc(recall, precision)
print(f"\n📊 AUPRC (Linear SVM): {auprc_lin:.4f}")
