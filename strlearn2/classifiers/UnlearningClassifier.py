import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
import numpy as np
from collections import deque


class UnlearningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        window_size=5,
        base_estimator=None,
        random_state=42,
        unlearning_rate = 0.2
    ):
        self.window_size = window_size
        self.random_state = random_state
        self.base_estimator = base_estimator
        self._is_initialized = False
        self.unlearning_rate = unlearning_rate
        #self.buffer_ = deque(maxlen=self.window_size)
        self.unbuffer_ = deque(maxlen=self.window_size)

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

    def _unlearn_chunk(self, X_old, y_old):
        """
        Approximate unlearning:
        ψ ← ψ − ∇L(DSk−L)
        """
        neg_weights = -np.ones(len(y_old)) * self.unlearning_rate
        print(neg_weights)
        self.model_.partial_fit(X_old, y_old, sample_weight=neg_weights)


    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, "buffer_"):
            self.buffer_ = deque(maxlen=1)
            self.k_ = 0                      # ← TU JEST KLUCZ
            self.classes_ = classes
            self._init_model()
            self._is_initialized = False
            self.train_times_ = []      # tylko uczenie
            self.memory_usage_ = []

        t_train_start = time.perf_counter()



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
        # k ≥ L → Unlearning + trening na najnowszym
        # ==================================================
        else:
            # === 1. UNLEARN najstarszego chunka ===
            X_old, y_old = self.buffer_[0]
            self._unlearn_chunk(X_old, y_old)

            # === 2. TRAIN na najnowszym chunku ===
            self.model_.partial_fit(X, y)

        # --- zapamiętaj chunk ---
        self.buffer_.append((X, y))

        mem = sum(
            c[0].nbytes + c[1].nbytes
            for c in self.buffer_
        )
        self.memory_usage_.append(mem)
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
