"""
Decision Tree Models for Tox21 SR-MMP
Uses PCA features from 01_data_loading_ecfp.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("=" * 60)
print("🌳 DECISION TREE MODELS")
print("=" * 60)

print("\n📂 Loading preprocessed data...")
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# =====================================================================
# 1) DECISION TREE with hyperparameter tuning
# =====================================================================
print("\n🌳 Decision Tree: hyperparameter search (max_depth, min_samples_split)...")

param_grid_dt = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_dt = GridSearchCV(
    dt,
    param_grid_dt,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)
print(f"✅ Best params (DT): {grid_dt.best_params_}")
print(f"✅ Best CV AUROC (DT): {grid_dt.best_score_:.4f}")

# Test performansı
best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)
y_proba_dt = best_dt.predict_proba(X_test)[:, 1]

test_auc_dt = roc_auc_score(y_test, y_proba_dt)
print(f"\n🧪 Decision Tree Test AUROC: {test_auc_dt:.4f}")

print("\n📊 Classification report (Decision Tree):")
print(classification_report(y_test, y_pred_dt, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp) if tp + fp > 0 else 0.0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (Decision Tree):")
print(f"   Sensitivity (TPR): {sens:.4f}")
print(f"   Specificity (TNR): {spec:.4f}")
print(f"   Precision (PPV):   {prec:.4f}")
print(f"   F1-Score:          {f1:.4f}")

fpr_dt, tpr_dt, thr_dt = roc_curve(y_test, y_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)
print(f"\n📈 ROC AUC (DT from fpr/tpr): {roc_auc_dt:.4f}")
print(f"   Thresholds: {len(thr_dt)}")

# Feature importance (bonus: hangi özellikler önemli?)
print(f"\n⭐ Feature Importance (Top 10):")
feature_importance = best_dt.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]
for i, idx in enumerate(top_indices, 1):
    print(f"   {i:2d}. Feature {idx:3d}: {feature_importance[idx]:.4f}")

print("\n" + "=" * 60)
print("✅ DECISION TREE MODELS DONE")
print("=" * 60)
