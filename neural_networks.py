import numpy as np
from metrics import one_hot, accuracy, cross_entropy, confusion_matrix, report_final_metrics as metrics_report, plot_training_history as plot_history

class NeuralNetwork:
    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def initialize_parameters(self, layer_sizes):
        self.params = {}
        for l in range(1, len(layer_sizes)):
            self.params[f"W{l}"] = np.random.randn(layer_sizes[l-1], layer_sizes[l]) * np.sqrt(2. / layer_sizes[l-1])
            self.params[f"b{l}"] = np.zeros((1, layer_sizes[l]))
        return self.params

    def forward(self, X, params=None):
        if params is None:
            params = self.params
        cache = {"A0": X}
        L = len(params) // 2
        for l in range(1, L):
            Z = np.dot(cache[f"A{l-1}"], params[f"W{l}"]) + params[f"b{l}"]
            A = self.relu(Z)
            cache[f"Z{l}"], cache[f"A{l}"] = Z, A
        ZL = np.dot(cache[f"A{L-1}"] , params[f"W{L}"]) + params[f"b{L}"]
        AL = self.softmax(ZL)
        cache[f"Z{L}"], cache[f"A{L}"] = ZL, AL
        return AL, cache

    def backward(self, X, Y, params=None, cache=None):
        if params is None:
            params = self.params
        grads = {}
        m = X.shape[0]
        L = len(params) // 2
        dZL = cache[f"A{L}"] - Y
        grads[f"dW{L}"] = np.dot(cache[f"A{L-1}"].T, dZL) / m
        grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True) / m
        for l in reversed(range(1, L)):
            dA = np.dot(dZL, params[f"W{l+1}"].T)
            dZ = dA * self.relu_derivative(cache[f"Z{l}"])
            grads[f"dW{l}"] = np.dot(cache[f"A{l-1}"].T, dZ) / m
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
            dZL = dZ
        return grads

    def update_parameters(self, params, grads, lr):
        for key in params:
            params[key] -= lr * grads["d" + key]
        return params

    def train_model(self, X_train, y_train, X_val, y_val, layer_sizes, epochs, lr_init,
                    lr_min=1e-5, batch_size=None, use_l2=False, lambda_l2=0.0,
                    use_early_stopping=False, patience=10,
                    use_lr_decay=False, decay_type="linear",
                    use_adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.initialize_parameters(layer_sizes)
        y_train_oh = one_hot(y_train, layer_sizes[-1])
        y_val_oh = one_hot(y_val, layer_sizes[-1])

        if use_adam:
            opt_mode = "adam"
            m = {key: np.zeros_like(val) for key, val in self.params.items()}
            v = {key: np.zeros_like(val) for key, val in self.params.items()}
            t = 0
        elif batch_size:
            opt_mode = "sgd"
        else:
            opt_mode = "gd"



        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float('inf')
        best_params = None
        no_improve = 0

        for epoch in range(epochs):
            if use_lr_decay:
                if decay_type == "linear":
                    lr = max(lr_init * (1 - epoch / epochs), lr_min)
                elif decay_type == "exponential":
                    gamma = (lr_min / lr_init) ** (1 / epochs)
                    lr = lr_init * (gamma ** epoch)
                else:
                    lr = lr_init
            else:
                lr = lr_init

            if opt_mode in ["adam", "sgd"]:
                idx = np.random.permutation(X_train.shape[0])
                X_train_shuff = X_train[idx]
                y_train_oh_shuff = y_train_oh[idx]
                for i in range(0, X_train.shape[0], batch_size):
                    X_batch = X_train_shuff[i:i+batch_size]
                    y_batch = y_train_oh_shuff[i:i+batch_size]
                    y_pred, cache = self.forward(X_batch, self.params)
                    grads = self.backward(X_batch, y_batch, self.params, cache)

                    if use_l2:
                        for l in range(1, len(layer_sizes)):
                            grads[f"dW{l}"] += lambda_l2 * self.params[f"W{l}"]

                    if opt_mode == "adam":
                        t += 1
                        for key in self.params:
                            m[key] = beta1 * m[key] + (1 - beta1) * grads["d" + key]
                            v[key] = beta2 * v[key] + (1 - beta2) * (grads["d" + key] ** 2)
                            m_hat = m[key] / (1 - beta1 ** t)
                            v_hat = v[key] / (1 - beta2 ** t)
                            self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
                    else:
                        self.params = self.update_parameters(self.params, grads, lr)
            else:
                y_pred, cache = self.forward(X_train)
                grads = self.backward(X_train, y_train_oh, self.params, cache)
                self.params = self.update_parameters(self.params, grads, lr)

            y_pred_train, _ = self.forward(X_train, self.params)
            y_pred_val, _ = self.forward(X_val, self.params)
            loss_train = cross_entropy(y_train_oh, y_pred_train)
            loss_val = cross_entropy(y_val_oh, y_pred_val)
            acc_train = accuracy(y_train_oh, y_pred_train)
            acc_val = accuracy(y_val_oh, y_pred_val)

            history["train_loss"].append(loss_train)
            history["val_loss"].append(loss_val)
            history["train_acc"].append(acc_train)
            history["val_acc"].append(acc_val)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Época {epoch+1}/{epochs} - lr: {lr:.5f} - Train Loss: {loss_train:.4f} - Val Loss: {loss_val:.4f} - Val Acc: {acc_val:.4f}")

            if use_early_stopping:
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_params = {k: v.copy() for k, v in self.params.items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping en época {epoch+1}")
                        break

        if use_early_stopping and best_params:
            self.params = best_params

        self.history = history
        return self.params, history

    def report_final_metrics(self, X, y, params=None, set_name="Train"):
        if params is None:
            params = self.params

        def forward_fn(X_input):
            probs, _ = self.forward(X_input, params)
            return probs, None

        num_classes = self.params[list(self.params.keys())[-1]].shape[1]
        metrics_report(X, y, forward_fn, num_classes, set_name=set_name)

    def plot_training_history(self, history, nombre_modelo="Modelo"):
        plot_history(history, nombre_modelo=nombre_modelo)

