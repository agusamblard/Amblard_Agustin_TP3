import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def one_hot(y, num_classes):
    return np.eye(num_classes)[y.astype(int)]

def accuracy(y_true_oh, y_pred_probs):
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_true_oh, axis=1)
    return np.mean(y_pred_labels == y_true_labels)

def cross_entropy(y_true_oh, y_pred_probs):
    return -np.mean(np.sum(y_true_oh * np.log(np.clip(y_pred_probs, 1e-12, 1. - 1e-12)), axis=1))

def confusion_matrix(y_true, y_pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred_labels):
        cm[int(t), int(p)] += 1
    return cm

def plot_training_history(history, nombre_modelo="Modelo"):

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{nombre_modelo} - Evolución de Cross-Entropy")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"{nombre_modelo} - Evolución de Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def report_final_metrics(X, y, forward_fn, num_classes, set_name="Train"):
    y_pred_probs, _ = forward_fn(X)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_oh = one_hot(y, num_classes)

    acc = accuracy(y_true_oh, y_pred_probs)
    loss = cross_entropy(y_true_oh, y_pred_probs)
    cm = confusion_matrix(y, y_pred_labels, num_classes)

    print(f"\nMétricas finales para {set_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Cross-Entropy Loss: {loss:.4f}")
    print(f"Matriz de Confusión ({set_name}):")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, cmap="Blues" if set_name == "Train" else "Oranges", fmt="d")
    plt.title(f"Matriz de Confusión - {set_name}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()












