"""
Ensemble Methods for Tox21 SR-MMP
Uses PCA features from 01_data_loading_ecfp.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("=" * 60)
print("🌲 ENSEMBLE METHODS: Random Forest & Gradient Boosting")
print("=" * 60)

print("\n📂 Loading preprocessed data...")
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================================
# 1) RANDOM FOREST
# =====================================================================
print("\n🌲 Random Forest (n_estimators=100) with 5-fold CV...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"✅ CV AUROC: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores_rf]}")

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

test_auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"\n🧪 Random Forest Test AUROC: {test_auc_rf:.4f}")

print("\n📊 Classification report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
sens_rf = tp / (tp + fn)
spec_rf = tn / (tn + fp)
prec_rf = tp / (tp + fp) if tp + fp > 0 else 0.0
f1_rf = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (Random Forest):")
print(f"   Sensitivity (TPR): {sens_rf:.4f}")
print(f"   Specificity (TNR): {spec_rf:.4f}")
print(f"   Precision (PPV):   {prec_rf:.4f}")
print(f"   F1-Score:          {f1_rf:.4f}")

fpr_rf, tpr_rf, thr_rf = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
print(f"\n📈 ROC AUC (RF from fpr/tpr): {roc_auc_rf:.4f}")

# AUPRC for Random Forest
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, y_proba_rf)
auprc_rf = auc(recall, precision)
print(f"\n📊 AUPRC (Random Forest): {auprc_rf:.4f}")


# =====================================================================
# 2) GRADIENT BOOSTING
# =====================================================================
print("\n\n⚡ Gradient Boosting (n_estimators=100) with 5-fold CV...")

gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

cv_scores_gb = cross_val_score(gb, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"✅ CV AUROC: {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores_gb]}")

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

test_auc_gb = roc_auc_score(y_test, y_proba_gb)
print(f"\n🧪 Gradient Boosting Test AUROC: {test_auc_gb:.4f}")

print("\n📊 Classification report (Gradient Boosting):")
print(classification_report(y_test, y_pred_gb, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_gb).ravel()
sens_gb = tp / (tp + fn)
spec_gb = tn / (tn + fp)
prec_gb = tp / (tp + fp) if tp + fp > 0 else 0.0
f1_gb = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (Gradient Boosting):")
print(f"   Sensitivity (TPR): {sens_gb:.4f}")
print(f"   Specificity (TNR): {spec_gb:.4f}")
print(f"   Precision (PPV):   {prec_gb:.4f}")
print(f"   F1-Score:          {f1_gb:.4f}")

fpr_gb, tpr_gb, thr_gb = roc_curve(y_test, y_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)
print(f"\n📈 ROC AUC (GB from fpr/tpr): {roc_auc_gb:.4f}")

print("\n" + "=" * 60)
print("✅ ENSEMBLE METHODS DONE")
print("=" * 60)

# AUPRC for Gradient Boosting (sonuna ekle)
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, y_proba_gb)
auprc_gb = auc(recall, precision)
print(f"\n📊 AUPRC (Gradient Boosting): {auprc_gb:.4f}")
