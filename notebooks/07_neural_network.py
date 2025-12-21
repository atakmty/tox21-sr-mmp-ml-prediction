"""
Neural Network (PyTorch) for Tox21 SR-MMP
Uses PCA features from 01_data_loading_ecfp.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc
)

print("=" * 60)
print("🧠 NEURAL NETWORK (PyTorch Deep Learning)")
print("=" * 60)

print("\n📂 Loading preprocessed data...")
X_train = np.load('X_train_pca.npy').astype(np.float32)
X_test = np.load('X_test_pca.npy').astype(np.float32)
y_train = np.load('y_train.npy').astype(np.float32)
y_test = np.load('y_test.npy').astype(np.float32)

print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test).view(-1, 1)

# =====================================================================
# BUILD NEURAL NETWORK
# =====================================================================
print("\n🔨 Building Neural Network architecture...")

class ToxicityNN(nn.Module):
    def __init__(self):
        super(ToxicityNN, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

model = ToxicityNN()
print(f"✅ Model architecture created")
print(f"   Input: 100 (PCA features)")
print(f"   Layer 1: 64 neurons, relu, dropout 0.3")
print(f"   Layer 2: 32 neurons, relu, dropout 0.2")
print(f"   Layer 3: 16 neurons, relu, dropout 0.2")
print(f"   Output: 1 neuron, sigmoid (binary classification)")

# =====================================================================
# TRAIN
# =====================================================================
print("\n⚙️  Training setup...")

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 20
batch_size = 32

print(f"🎓 Training Neural Network ({epochs} epochs, batch_size={batch_size})...")

model.train()
for epoch in range(epochs):
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print(f"✅ Training Complete")

# =====================================================================
# EVALUATE ON TEST SET
# =====================================================================
print("\n🧪 Evaluating on test set...")

model.eval()
with torch.no_grad():
    y_proba_nn = model(X_test_tensor).numpy().flatten()

y_pred_nn = (y_proba_nn >= 0.5).astype(int)

test_auc_nn = roc_auc_score(y_test, y_proba_nn)
print(f"✅ Neural Network Test AUROC: {test_auc_nn:.4f}")

print("\n📊 Classification report (Neural Network):")
print(classification_report(y_test, y_pred_nn, target_names=['Non-toxic', 'Toxic']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_nn).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp) if tp + fp > 0 else 0.0
f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

print("\n🔍 Detailed metrics (Neural Network):")
print(f"   Sensitivity (TPR): {sens:.4f}")
print(f"   Specificity (TNR): {spec:.4f}")
print(f"   Precision (PPV):   {prec:.4f}")
print(f"   F1-Score:          {f1:.4f}")

fpr_nn, tpr_nn, thr_nn = roc_curve(y_test, y_proba_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)
print(f"\n📈 ROC AUC (NN from fpr/tpr): {roc_auc_nn:.4f}")
print(f"   Thresholds: {len(thr_nn)}")

print("\n" + "=" * 60)
print("✅ NEURAL NETWORK DONE")
print("=" * 60)

# AUPRC for Neural Network
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_test, y_proba_nn)
auprc_nn = auc(recall, precision)
print(f"\n📊 AUPRC (Neural Network): {auprc_nn:.4f}")
