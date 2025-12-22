# Comparative Machine Learning Analysis for Mitochondrial Membrane Potential Toxicity Prediction Using Tox21 Dataset

**Developed by:** Ata Kamutay  
**Date:** December 2025  

---

## Overview

This repository contains a comprehensive machine learning project that systematically evaluates **8 different algorithms** for predicting mitochondrial membrane potential (MMP) toxicity using the Tox21 dataset. The project implements core machine learning techniques and achieves competitive performance benchmarks for computational toxicology applications.

**Key Achievement:** RBF SVM achieved **AUROC 0.8682** with excellent balanced accuracy (0.7690), representing a **19.3% improvement** over baseline KNN and demonstrating strong predictive performance for drug toxicity screening.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/atakmty/tox21-sr-mmp-ml-prediction.git
cd tox21-sr-mmp-ml-prediction

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, sklearn, rdkit, pandas; print('✓ All packages installed!')"
```

### Run Models

```bash
cd notebooks

# Run individual models
python 02_baseline_knn.py                    # K-Nearest Neighbors baseline
python 03_svm_models.py                      # RBF & Linear SVM
python 04_decision_trees.py                  # Decision Tree with hyperparameter tuning
python 05_naive_bayes.py                     # Gaussian Naive Bayes
python 06_ensemble_methods.py                # Random Forest & Gradient Boosting
python 07_neural_network.py                  # PyTorch Neural Network
python 08_final_results_summary.py           # Generate final comparison table
python 09_auprc_analysis.py                  # AUPRC analysis for imbalanced data
```

---

## Project Structure

```
tox21-sr-mmp-ml-prediction/
├── data/
│   ├── sr-mmp-train.smiles                  # Training SMILES (7,320 compounds)
│   └── sr-mmp-test.smiles                   # Test SMILES (238 compounds)
├── notebooks/
│   ├── 01_data_loading_ecfp.py              # Data loading & PCA preprocessing
│   ├── 02_baseline_knn.py                   # K-Nearest Neighbors (k=5)
│   ├── 03_svm_models.py                     # RBF SVM & Linear SVM with GridSearchCV
│   ├── 04_decision_trees.py                 # Decision Tree with hyperparameter tuning
│   ├── 05_naive_bayes.py                    # Gaussian Naive Bayes classifier
│   ├── 06_ensemble_methods.py               # Random Forest & Gradient Boosting
│   ├── 07_neural_network.py                 # PyTorch deep neural network
│   ├── 08_final_results_summary.py          # Final model comparison & ranking
│   └── 09_auprc_analysis.py                 # Precision-recall analysis
├── results/
│   └── final_model_comparison_complete_with_auprc.csv  # Final metrics table
├── README.md                                 # This file
├── requirements.txt                         # Python dependencies
└── .gitignore                               # Git ignore rules
```

---

## Dataset

**Source:** Tox21 SR-MMP (Mitochondrial Membrane Potential)  
**Provider:** U.S. EPA, NIH, FDA Tox21 Initiative  

**Dataset Statistics:**
- **Total Compounds:** 7,558
- **Training Set:** 7,320 compounds (80%)
- **Test Set:** 238 compounds (20%)
- **Class Distribution:** 15.6% toxic (active), 84.4% non-toxic (inactive)
- **Features:** Morgan ECFP (radius=2, 1024-bit vectors)
- **After Preprocessing:** 100 PCA components capturing 57.2% variance

**Data Handling:**
- Stratified sampling to maintain class proportions
- Class weighting (`class_weight='balanced'`) for imbalanced classification
- Evaluation metrics prioritized AUROC, Balanced Accuracy, AUPRC over simple accuracy

---

## Methodology

### Feature Engineering

**Molecular Representation: Morgan ECFP**
- Radius: 2 (ECFP4 equivalent)
- Bit vector length: 1024
- Tool: RDKit chemistry toolkit
- Captures circular atom neighborhoods through iterative hashing for structural encoding

**Dimensionality Reduction: Principal Component Analysis**
- Components: 100
- Explained variance: 57.2%
- Purpose: Reduce 1024-dimensional feature space while preserving signal

### Machine Learning Models

#### Baseline Models
- **K-Nearest Neighbors (KNN):** k=5, Euclidean distance metric
- **Gaussian Naive Bayes:** Probabilistic baseline classifier

#### Single Classifiers
- **Decision Trees:** GridSearchCV hyperparameter optimization
  - Tuned parameters: max_depth [5, 10, 15, 20], min_samples_split [2, 5, 10]
  
- **Support Vector Machines (SVM):**
  - RBF Kernel: C=1, gamma='scale' (GridSearchCV optimized) ⭐ WINNER
  - Linear Kernel: For comparison and sensitivity analysis
  
- **Random Forest:** 100 decision trees (Bagging ensemble)
  - max_depth=10, min_samples_split=5

- **Gradient Boosting:** 100 sequential estimators
  - max_depth=5, learning_rate=0.1

#### Deep Learning
- **Neural Network (PyTorch):**
  - Architecture: 100 → 64 (ReLU, 30% Dropout) → 32 (ReLU, 20% Dropout) → 16 (ReLU, 20% Dropout) → 1 (Sigmoid)
  - Training: 20 epochs, batch_size=32, Adam optimizer (lr=0.001)
  - Loss function: Binary Cross-Entropy

### Evaluation Strategy

**Cross-Validation:** 5-fold stratified cross-validation on training data  
**Test Evaluation:** Hold-out 20% test set for unbiased performance assessment

**Evaluation Metrics:**
1. **AUROC** - Primary metric for overall discrimination ability
2. **AUPRC** - Secondary metric for minority class (toxic) performance
3. **Balanced Accuracy** - Fair assessment metric for imbalanced data
4. **Sensitivity** - True positive rate (ability to detect toxic compounds)
5. **Specificity** - True negative rate (ability to avoid false alarms)
6. **Precision** - Positive predictive value
7. **F1-Score** - Harmonic mean emphasizing minority class

---

## Results

### Final Model Comparison

| Rank | Model | Test AUROC | AUPRC | Sensitivity | Specificity | Precision | F1-Score | Balanced Accuracy |
|------|-------|-----------|-------|-------------|-------------|-----------|----------|-------------------|
| 🥇 1 | **RBF SVM** | **0.8682** | 0.5687 | **0.6579** | **0.8800** | 0.5102 | **0.5747** | **0.7690** |
| 🥈 2 | Gradient Boosting | 0.8513 | 0.5070 | 0.3158 | 0.9700 | 0.6667 | 0.4286 | 0.6429 |
| 🥉 3 | Neural Network | 0.8403 | **0.6409** | 0.2895 | 0.9850 | 0.7857 | 0.4231 | 0.6372 |
| 4 | Linear SVM | 0.8348 | 0.5173 | 0.7632 | 0.7700 | 0.3867 | 0.5133 | 0.7666 |
| 5 | Random Forest | 0.7814 | 0.4370 | 0.2105 | 0.9700 | 0.5714 | 0.3077 | 0.5902 |
| 6 | Naive Bayes | 0.7403 | 0.3155 | 0.6053 | 0.7500 | 0.3151 | 0.4144 | 0.6776 |
| 7 | KNN | 0.7243 | 0.4355 | 0.2895 | 0.9500 | 0.5238 | 0.3729 | 0.6198 |
| 8 | Decision Tree | 0.6630 | 0.3280 | 0.5263 | 0.6500 | 0.2222 | 0.3125 | 0.5881 |

### Key Findings

**🥇 RBF SVM - Optimal Model:**
- Highest test AUROC: 0.8682 (excellent discrimination)
- Best balanced accuracy: 0.7690 (optimal sensitivity-specificity balance)
- Best F1-Score: 0.5747 (best harmonic mean for minority class)
- **Performance:** Catches 66% of toxic compounds while avoiding 88% of false alarms
- **Application:** Ideal for practical drug screening where balanced error rates matter

**🥈 Neural Network - Highest AUPRC:**
- Highest AUPRC: 0.6409 (best precision-recall balance for toxic compounds)
- Very high specificity: 0.9850 (fewer false positives)
- Trade-off: Low sensitivity (0.2895) - misses more toxic compounds
- **Application:** Better for precision-focused scenarios where false positives are costly

**Performance Aggregate Statistics:**
- Mean AUROC: 0.7879 ± 0.0731
- Mean AUPRC: 0.4687 ± 0.1127
- Mean Balanced Accuracy: 0.6614 ± 0.0762
- AUROC Range: 0.6630–0.8682 (19.3% improvement from worst to best)

---

## Installation Guide

### System Requirements

- Python 3.9 or higher (tested on Python 3.14)
- 4+ GB RAM (for RDKit and model training)
- ~5 minutes for full installation

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/atakmty/tox21-sr-mmp-ml-prediction.git
cd tox21-sr-mmp-ml-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "
import numpy as np
import pandas as pd
import sklearn
import torch
import rdkit
print('✓ All packages installed successfully!')
print(f'  - NumPy: {np.__version__}')
print(f'  - Pandas: {pd.__version__}')
print(f'  - scikit-learn: {sklearn.__version__}')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - RDKit: {rdkit.__version__}')
"

# 5. Run analysis
cd notebooks
python 01_data_loading_ecfp.py
```

### Troubleshooting

**RDKit Installation Issues on Windows:**
```bash
# If standard pip installation fails, use conda
conda install -c conda-forge rdkit
```

**PyTorch GPU Support (Optional):**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Memory Issues:**
```bash
# Reduce batch size in notebook scripts if running on low RAM
# Edit batch_size parameter in ensemble_methods.py and neural_network.py
```

---

## Dependencies

See `requirements.txt` for complete list. Critical packages:

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing and array operations |
| pandas | Data manipulation and DataFrame processing |
| scikit-learn | Machine learning models and evaluation |
| torch | Neural network framework (PyTorch) |
| rdkit | Molecular fingerprints and SMILES parsing |
| matplotlib | Visualization and plotting (optional) |
| seaborn | Statistical visualization (optional) |

---

## Project Highlights

✅ **Comprehensive Algorithm Comparison:** 8 different machine learning approaches systematically evaluated  
✅ **Imbalanced Data Handling:** Proper techniques for 15.6% minority class  
✅ **Hyperparameter Optimization:** GridSearchCV for kernel-based and tree-based models  
✅ **Multiple Evaluation Metrics:** AUROC, AUPRC, Balanced Accuracy, F1-Score for comprehensive assessment  
✅ **Production-Ready Code:** Well-documented, modular, and reproducible implementations  
✅ **Clinical Relevance:** Results applicable to pharmaceutical drug development pipelines  
✅ **Deep Learning Integration:** PyTorch neural network alongside classical ML methods

---

## Scientific Applications

This predictive model can accelerate pharmaceutical and chemical research by:

1. **Early Toxicity Screening:** Computationally identify mitochondrial toxic compounds before expensive laboratory testing
2. **Cost Reduction:** Computational prediction 10–100× faster than in vitro/in vivo experiments
3. **Animal Testing Reduction:** Serve as computational filter to reduce animal studies
4. **Drug Development Efficiency:** Enable screening of larger chemical libraries with limited resources

**Potential Impact:** Drug development costs $2.6B on average; computational filtering could reduce by 10–15% through early elimination of toxic candidates


---

## Contact & Support

**Developer:** Ata Kamutay  
**GitHub:** https://github.com/atakmty  
**Repository Issues:** https://github.com/atakmty/tox21-sr-mmp-ml-prediction/issues

For questions or suggestions, please open an issue on the GitHub repository.

---

## License

This project is made available for research and educational purposes.

---

## Data Sources & Acknowledgments

- **Tox21 Dataset:** U.S. Environmental Protection Agency (EPA), National Institutes of Health (NIH), Food and Drug Administration (FDA)
- **RDKit:** Open-source cheminformatics toolkit
- **PyTorch & scikit-learn:** Community-driven open-source machine learning libraries
- **Tox21 Publications:** 
  - Huang et al. (2016). *Nature Communications*, 7, 10425
  - Wu et al. (2021). *Chemical Research in Toxicology*, 34(2), 541-549
  - Bringezu et al. (2021). *Computational Toxicology*, 18, 100166

---

**Last Updated:** December 21, 2025