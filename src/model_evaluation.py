"""
Model Evaluation Utilities
Tox21 SR-MMP Toxicity Prediction
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.results = {}

    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Full evaluation metrics

        Parameters:
        -----------
        y_true : array
            Ground truth labels
        y_pred : array
            Predicted labels (0/1)
        y_pred_proba : array, optional
            Predicted probabilities [0, 1]

        Returns:
        --------
        dict : All metrics
        """
        results = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
        }

        if y_pred_proba is not None:
            results['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)

        self.results = results
        return results

    def print_results(self):
        """Pretty print results"""
        print("\n" + "=" * 50)
        print(f"✅ {self.model_name} RESULTS")
        print("=" * 50)
        for metric, value in self.results.items():
            print(f"{metric:.<30} {value:.4f}")
        print("=" * 50 + "\n")

    def confusion_matrix_analysis(self, y_true, y_pred):
        """Analyze confusion matrix"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn)  # True Positive Rate
        specificity = tn / (tn + fp)  # True Negative Rate
        ppv = tp / (tp + fp)  # Precision
        npv = tn / (tn + fn)  # Negative Predictive Value

        print(f"\n{'Confusion Matrix Analysis':^50}")
        print("-" * 50)
        print(f"TP: {tp:5d} | FP: {fp:5d}")
        print(f"FN: {fn:5d} | TN: {tn:5d}")
        print("-" * 50)
        print(f"Sensitivity (TPR): {sensitivity:.4f}  [% of toxic correctly detected]")
        print(f"Specificity (TNR): {specificity:.4f}  [% of non-toxic correctly detected]")
        print(f"PPV (Precision):   {ppv:.4f}  [% of predicted toxic that are correct]")
        print(f"NPV:               {npv:.4f}  [% of predicted non-toxic that are correct]")

        return {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Sensitivity': sensitivity, 'Specificity': specificity,
            'PPV': ppv, 'NPV': npv
        }

    def roc_analysis(self, y_true, y_pred_proba):
        """ROC curve analysis"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        print(f"\n{'ROC Curve Analysis':^50}")
        print("-" * 50)
        print(f"AUC: {roc_auc:.4f}")
        print(f"Number of thresholds: {len(thresholds)}")

        return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'thresholds': thresholds}

    def pr_analysis(self, y_true, y_pred_proba):
        """Precision-Recall curve analysis"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        print(f"\n{'Precision-Recall Analysis':^50}")
        print("-" * 50)
        print(f"PR-AUC: {pr_auc:.4f}")

        return {'precision': precision, 'recall': recall, 'pr_auc': pr_auc}


def compare_models(results_dict):
    """
    Compare multiple models

    Parameters:
    -----------
    results_dict : dict
        {model_name: {metric: value}}
    """
    print("\n" + "=" * 70)
    print(f"{'MODEL COMPARISON':^70}")
    print("=" * 70)

    # Get all metrics
    metrics = list(next(iter(results_dict.values())).keys())

    # Print header
    print(f"{'Model':<20}", end='')
    for metric in metrics:
        print(f"{metric:>12}", end='')
    print()
    print("-" * 70)

    # Print results
    for model_name, metrics_dict in results_dict.items():
        print(f"{model_name:<20}", end='')
        for metric in metrics:
            value = metrics_dict.get(metric, 0)
            print(f"{value:>12.4f}", end='')
        print()

    print("=" * 70 + "\n")


def save_results_csv(results_dict, filename='results.csv'):
    """Save results to CSV"""
    import pandas as pd

    df = pd.DataFrame(results_dict).T
    df.to_csv(filename)
    print(f"✅ Saved: {filename}")
