import numpy as np
import time
from collections import deque
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier


class NewMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Streaming MLPClassifier with DeltaGrad unlearning (sklearn-compatible)
    """

    
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        lr=0.01,
        ulr=0.01,
        unlearn_steps=1,
        window_size=5,
        random_state=None
    ):
        self.lr = lr
        self.ulr = ulr
        self.unlearn_steps = unlearn_steps
        self.window_size = window_size 

        self.model = MLPClassifier(
            hidden_layer_sizes=(100,),
            solver="adam",
            max_iter=200,
            warm_start=False,
            random_state=42
        )

        self.buffer = deque(maxlen=window_size)
        self.classes_ = None
        self.k = 0

        self.train_times_ = []
        self.memory_usage_ = []
    
        # --------------------------------------------------
    # Forward / Backward
    # --------------------------------------------------

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X):
        activations = [X]
        pre_activations = []

        for W, b in zip(self.model.coefs_, self.model.intercepts_):
            z = activations[-1] @ W + b
            pre_activations.append(z)

            if W is self.model.coefs_[-1]:
                a = self._softmax(z)
            else:
                a = self._relu(z)

            activations.append(a)

        return activations, pre_activations

    def _backward(self, activations, pre_activations, y):
        y_onehot = np.zeros((len(y), self.classes_.size))
        y_onehot[np.arange(len(y)), y] = 1

        grads_W = []
        grads_b = []

        delta = activations[-1] - y_onehot  # dL/dz softmax

        for i in reversed(range(len(self.model.coefs_))):
            a_prev = activations[i]
            dW = a_prev.T @ delta / len(y)
            db = np.mean(delta, axis=0)

            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.model.coefs_[i].T) * self._relu_grad(pre_activations[i - 1])

        return grads_W, grads_b

    # --------------------------------------------------
    # Core logic
    # --------------------------------------------------

    def _unlearn(self, X_forget, y_forget):
        for _ in range(self.unlearn_steps):
            activations, pre_acts = self._forward(X_forget)
            dW, db = self._backward(activations, pre_acts, y_forget)

            # ODUCZANIE = + gradient
            for i in range(len(self.model.coefs_)):
                self.model.coefs_[i] += self.lr * dW[i]
                self.model.intercepts_[i] += self.lr * db[i]


    # ==================================================
    # Partial fit
    # ==================================================
    def partial_fit(self, X, y, classes=None):
        t0 = time.perf_counter()
        # inicjalizacja klas (tylko raz)
        if self.k == 0:
            self.classes_ = np.unique(y) if classes is None else classes

        # k < L → tylko Train
        if self.k < self.window_size:
            self.model.partial_fit(X, y, classes=self.classes_)

        # k ≥ L → Unlearn(DS_{k-L}) + Train(DS_k)
        else:
            X_forget, y_forget = self.buffer[0]
            self._unlearn(X_forget, y_forget)
            self.model.partial_fit(X, y)

        # zapamiętaj DS_k
        self.buffer.append((X.copy(), y.copy()))
        self.k += 1

        # --- LOGGING ---
        self.train_times_.append(time.perf_counter() - t0)
        self.memory_usage_.append(
            sum(dW.nbytes + db.nbytes for dW, db in self.buffer)
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
