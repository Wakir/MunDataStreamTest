import numpy as np
from collections import deque
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from ..ensembles.base import StreamingEnsemble


class SlidingWindowSEA(StreamingEnsemble):
    """
    SEA with Sliding Window mechanism.

    Trains a single model on the last L data chunks.
    After exceeding window size, the model deletes ensemble with the oldest classifier.
    """

    def __init__(self, base_estimator=None, n_estimators=10, metric=accuracy_score):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.metric = metric

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Append new estimator
        self.ensemble_.append(clone(self.base_estimator).fit(self.X_, self.y_))

        # Remove the oldest when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            del self.ensemble_[0]
        return self
