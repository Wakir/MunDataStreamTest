import numpy as np
from collections import deque
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from ..ensembles.base import StreamingEnsemble


class SlidingWindowSEA2(StreamingEnsemble):
    """
    SEA with Sliding Window mechanism.

    Trains a single model on the last L data chunks.
    After exceeding window size, the model is reset and retrained.
    """

    def __init__(self, base_estimator=None, n_estimators=10, metric=accuracy_score):
        super().__init__(base_estimator, n_estimators)
        self.buffer_X = []
        self.buffer_y = []
        self.k = 0
        self.metric = metric

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self
        # Dodaj chunk do okna
        self.buffer_X.append(X)
        self.buffer_y.append(y)

        # Jeśli przekroczono długość okna → usuń najstarszy
        if len(self.buffer_X) > self.n_estimators:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        # Zresetuj model
        self.ensemble_ = []
        model = clone(self.base_estimator)

        # Połącz dane z okna
        X_train = np.vstack(self.buffer_X)
        y_train = np.hstack(self.buffer_y)

        # Trening
        model.fit(X_train, y_train)
        self.ensemble_.append(model)

        self.k += 1
        return self
