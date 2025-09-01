import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from metrics import one_hot, accuracy, cross_entropy, confusion_matrix, report_final_metrics


class NeuralNetworkPytorch:
    def __init__(
        self, x_train, y_train, x_val, y_val,
        architecture_fn, learning_rate, lambda_l2,
        epochs, batch_size
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y_val, dtype=torch.long).to(self.device)

        self.train_loader = DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=batch_size, shuffle=True)

        self.model = architecture_fn().to(self.device)
        self._apply_he_initialization()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

        self.epochs = epochs

    def _apply_he_initialization(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def train(self):
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()

            val_loss = self._evaluate_val_loss()
            train_loss = self._evaluate_loss_on_set(self.x_train, self.y_train)
            train_acc = self._evaluate_accuracy_on_set(self.x_train, self.y_train)
            val_acc = self._evaluate_accuracy_on_set(self.x_val, self.y_val)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if epoch % 50 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        train_loss = self._evaluate_loss_on_set(self.x_train, self.y_train)
        val_loss = self._evaluate_loss_on_set(self.x_val, self.y_val)
        self.last_train_val_losses = (train_loss, val_loss)

    def _evaluate_val_loss(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.x_val)
            loss = self.criterion(outputs, self.y_val)
        return loss.item()

    def _evaluate_loss_on_set(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        return loss.item()

    def _evaluate_accuracy_on_set(self, X, y):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            preds = torch.argmax(outputs, dim=1)
            return (preds == y).float().mean().item()

    def forward(self, X, params=None):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs, None

    def report_final_metrics(self, X, y, params=None, set_name="Train"):
        if params is None:
            params = self.model.state_dict()

        def forward_fn(X_input):
            return self.forward(X_input, params)

        num_classes = self.model[-1].out_features if isinstance(self.model, nn.Sequential) else self.forward(X)[0].shape[1]
        report_final_metrics(X, y, forward_fn, num_classes, set_name=set_name)

    def plot_training_history(self, nombre_modelo="Modelo"):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title(f"{nombre_modelo} - Cross Entropy")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.title(f"{nombre_modelo} - Accuracy")
        plt.xlabel("Época")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()
