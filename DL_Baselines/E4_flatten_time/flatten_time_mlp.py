import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
import optuna
import numpy as np
import pandas as pd
import os
import json

# Global parameters
num_epochs = 100
patience = 10
min_delta = 0.001
validation_ratio = 0.2
num_splits = 5
results_dir = "./Results"
os.makedirs(results_dir, exist_ok=True)

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# FlattenMLP class
class FlattenMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super(FlattenMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        # Flatten the time and feature dimensions
        x = x.view(x.size(0), -1)
        output = self.mlp(x)
        return output

# Objective function for Optuna
def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 4, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    # Initialize the model
    model = FlattenMLP(input_dim=time_steps * features, hidden_dim=hidden_dim, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train).squeeze(1)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze(1)
            val_loss = criterion(val_outputs, y_val).item()

            # Early stopping check
            if early_stopping.should_stop(val_loss):
                print(f"Early stopping at epoch {epoch + 1} with validation loss {val_loss:.4f}")
                break

    # Evaluation on validation set
    probabilities = torch.sigmoid(val_outputs).numpy()
    predicted = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_val.numpy(), predicted)
    roc_auc = roc_auc_score(y_val.numpy(), probabilities)
    sensitivity = recall_score(y_val.numpy(), predicted)
    cm = confusion_matrix(y_val.numpy(), predicted)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Store the model and metrics
    trial.set_user_attr("model", model)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("sensitivity", sensitivity)
    trial.set_user_attr("specificity", specificity)

    return val_loss

# Dictionary to store metrics for each split
metrics = {'accuracy': [], 'roc_auc': [], 'sensitivity': [], 'specificity': []}

# Loop through each data split
for split in range(1, num_splits + 1):
    # Load training data
    X = np.load(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/X_train_tensor.npy")
    y = pd.read_csv(f"../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/S{split}/y_train_tensor.csv")[['MR']].MR.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_ratio, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_val = torch.tensor(y_val).float()

    time_steps = X_train.shape[1]
    features = X_train.shape[2]

    # Optimize with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    best_model = best_trial.user_attrs.get("model")

    # Save results for the current split
    split_dir = os.path.join(results_dir, f"split_{split}")
    os.makedirs(split_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(split_dir, "best_model.ckpt"))

    # Save hyperparameters and metrics in a JSON file
    results = {
        "hyperparameters": study.best_params,
        "metrics": {
            "accuracy": best_trial.user_attrs['accuracy'],
            "roc_auc": best_trial.user_attrs['roc_auc'],
            "sensitivity": best_trial.user_attrs['sensitivity'],
            "specificity": best_trial.user_attrs['specificity']
        }
    }
    with open(os.path.join(split_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Store metrics
    for metric in metrics:
        metrics[metric].append(best_trial.user_attrs[metric])

    print(f"\nSplit {split} saved at {split_dir}")

# Summary of metrics across all splits
print("\nSummary of All Splits:")
summary = {}
for metric in metrics:
    mean_value = np.mean(metrics[metric])
    std_value = np.std(metrics[metric])
    print(f"{metric.capitalize()} - Mean: {mean_value:.4f}, Std: {std_value:.4f}")
    summary[metric] = {"mean": mean_value, "std": std_value}

# Save summary to a JSON file
summary_path = os.path.join(results_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"\nSummary of all splits saved at {summary_path}")
