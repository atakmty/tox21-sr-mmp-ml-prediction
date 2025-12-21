"""
Baseline KNN Model for Tox21 SR-MMP
Proposal Section 4
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("="*60)
print("🎯 BASELINE: K-Nearest Neighbors (KNN)")
print("="*60)

# Load preprocessed data
print("\n📂 Loading data...")
X_train_pca = np.load('X_train_pca.npy')
X_test_pca = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   X_train: {X_train_pca.shape}")
print(f"   X_test:  {X_test_pca.shape}")

# ============================================================================
# STEP 1: CROSS-VALIDATION (k=5)
# ============================================================================
print("\n🔄 Cross-Validation (k=5)...")

knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train_pca, y_train, cv=5, scoring='roc_auc')

print(f"✅ CV Complete!")
print(f"   AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# ============================================================================
# STEP 2: TEST SET EVALUATION
# ============================================================================
print("\n🧪 Test Set Evaluation...")

knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)
y_pred_proba = knn.predict_proba(X_test_pca)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ Test AUROC: {test_auc:.4f}")

# ============================================================================
# STEP 3: DETAILED METRICS
# ============================================================================
print("\n📊 Classification Metrics:")
print(classification_report(y_test, y_pred, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print(f"\n   Sensitivity (TPR): {sensitivity:.4f}")
print(f"   Specificity (TNR): {specificity:.4f}")
print(f"   Precision: {tp/(tp+fp):.4f}")
print(f"   F1-Score: {2*tp/(2*tp+fp+fn):.4f}")

# ============================================================================
# STEP 4: ROC CURVE
# ============================================================================
print("\n📈 ROC Curve Analysis...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"   ROC AUC: {roc_auc:.4f}")
print(f"   Number of thresholds: {len(thresholds)}")

print("\n" + "="*60)
print("✅ KNN BASELINE COMPLETE!")
print(f"   CV AUROC: {cv_scores.mean():.4f}")
print(f"   Test AUROC: {test_auc:.4f}")
print("="*60)
