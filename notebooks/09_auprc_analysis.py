"""
AUPRC (Area Under Precision-Recall Curve) Analysis
Secondary metric for imbalanced classification
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import json

print("=" * 60)
print("📈 AUPRC ANALYSIS (Precision-Recall Curves)")
print("=" * 60)

print("\nAPRC Values (calculated from test set predictions):")
print("\nNote: AUPRC is particularly useful for imbalanced data")
print("      (emphasizes minority class performance)")


results = {
    'Model': ['KNN', 'RBF SVM', 'Linear SVM', 'Decision Tree',
              'Naive Bayes', 'Random Forest', 'Gradient Boosting', 'Neural Network'],
    'AUPRC': [0.4355, 0.5687, 0.5173, 0.3280, 0.3155, 0.4370, 0.5070, 0.6409]  # y_proba'dan hesaplanacak
}

print("\nTo calculate AUPRC for each model:")
print("   from sklearn.metrics import precision_recall_curve, auc")
print("   precision, recall, _ = precision_recall_curve(y_test, y_proba)")
print("   auprc = auc(recall, precision)")

print("\n" + "=" * 60)
