import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
import numpy as np
from collections import deque


class SlidingWindowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        window_size=5,
        base_estimator=None,
        random_state=42,
    ):
        self.window_size = window_size
        self.random_state = random_state
        self.base_estimator = base_estimator
        self._is_initialized = False

    # ==================================================
    # 1. Inicjalizacja modelu ψ
    # ==================================================
    def _init_model(self):
        if self.base_estimator is None:
            self.model_ = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=0.01,
                max_iter=1,
                tol=None,
            )
        else:
            self.model_ = self.base_estimator
        self._is_initialized = False

    # ==================================================
    # 2. Partial fit (chunk-based)
    # ==================================================
    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, "buffer_"):
            self.buffer_ = deque(maxlen=self.window_size)
            self.k_ = 0
            self.classes_ = classes
            self._init_model()
            self.train_times_ = []      # tylko uczenie
            self.memory_usage_ = []

        t_train_start = time.perf_counter()

        # --- zapamiętaj chunk ---
        self.buffer_.append((X, y))
        mem = sum(
            c[0].nbytes + c[1].nbytes
            for c in self.buffer_
        )
        self.memory_usage_.append(mem)


        # ==================================================
        # k < L  → inkrementalne uczenie
        # ==================================================
        if self.k_ < self.window_size:
            if not self._is_initialized:
                self.model_.partial_fit(X, y, classes=self.classes_)
                self._is_initialized = True
            else:
                self.model_.partial_fit(X, y)

        # ==================================================
        # k ≥ L → RESET + trening od zera na oknie
        # ==================================================
        else:
            self._init_model()

            Xw = np.vstack([c[0] for c in self.buffer_])
            yw = np.hstack([c[1] for c in self.buffer_])

            self.model_.fit(Xw, yw)
            self._is_initialized = True

        t_train_end = time.perf_counter()
        self.train_times_.append(t_train_end - t_train_start)
        self.k_ += 1
        return self

    # ==================================================
    # 3. Predykcja
    # ==================================================
    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
