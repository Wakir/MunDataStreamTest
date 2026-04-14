import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from collections import deque

from sklearn.base import BaseEstimator, ClassifierMixin
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


# ======================================================
# Sliding Window ResNet
# ======================================================
class SlidingWindowPerceptron(BaseEstimator, ClassifierMixin):

    def __init__(self, window_size=5, lr=1e-3, epochs=10):

        self.window_size = window_size
        self.lr = lr
        self.epochs = epochs

        self.classes_ = np.arange(10)
        self._is_initialized = False

        self.train_times_ = []
        self.memory_usage_ = []

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = resnet18(num_classes=10).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr
        )

        self._is_initialized = True

        self.buffer = deque(maxlen=window_size)

        self.k = 0

        self.classes_ = None

        self.train_times_ = []
        self.memory_usage_ = []

    def _init_model(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = resnet18(num_classes=10).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr
        )

        self._is_initialized = True
    
    def _track_memory_usage(self):

        mem = 0

        for p in self.model.parameters():
            mem += p.nelement() * p.element_size()

            if p.grad is not None:
                mem += p.grad.nelement() * p.grad.element_size()

        if hasattr(self, "buffer_"):
            mem += sum(c[0].nbytes + c[1].nbytes for c in self.buffer_)

        if torch.cuda.is_available():
            mem += torch.cuda.memory_allocated(self.device)

        self.memory_usage_.append(mem)


    # --------------------------------------------------
    # DATA PREP (MNIST -> tensor)
    # --------------------------------------------------
    def _prepare_input(self, X):

        if isinstance(X, np.ndarray):

            # NHWC -> NCHW
            if X.ndim == 4:
                X = np.transpose(X, (0, 3, 1, 2))

            if X.shape[1] == 1:
                X = np.repeat(X, 3, axis=1)

            if X.ndim == 2 and X.shape[1] == 784:
                X = X.reshape(-1, 1, 28, 28)
                X = np.repeat(X, 3, axis=1)

        X = torch.tensor(X).float().to(self.device)

        return X
    
    def _prepare_target(self, y):

        return torch.tensor(y).long().to(self.device)

    # --------------------------------------------------
    # DATA CHUNK TRAIN
    # --------------------------------------------------

    def _train(self, X, y, epochs):

        X = self._prepare_input(X)
        y = self._prepare_target(y)

        dataset = TensorDataset(X, y)

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True
        )

        for _ in range(epochs):
            for xb, yb in loader:

                self.optimizer.zero_grad()

                out = self.model(xb)

                loss = self.loss_fn(out, yb)

                loss.backward()

                self.optimizer.step()

    # --------------------------------------------------
    # partial_fit (Sliding Window)
    # --------------------------------------------------
    def partial_fit(self, X, y, classes=None):

        if not hasattr(self, "buffer_"):
            self.buffer_ = deque(maxlen=self.window_size)
            self.k_ = 0

            if classes is not None:
                self.classes_ = classes
        print(self.k_)    
        t_start = time.perf_counter()

        if not self._is_initialized:
            self._init_model()

        # ==================================================
        # k < L → inremental training
        # ==================================================
        if self.k_ < self.window_size:

            self._train(X, y, self.epochs)
            self.buffer_.append((X, y))

        # ==================================================
        # k ≥ L → reset + training on whole window
        # ==================================================
        else:
            self.buffer_.append((X, y))
            Xw = np.vstack([c[0]for c in self.buffer_])
            yw = np.hstack([c[1] for c in self.buffer_])

            self._init_model()

            self._train(Xw, yw, self.epochs)

        t_end = time.perf_counter()

        self.train_times_.append(t_end - t_start)

        self._track_memory_usage()

        self.k_ += 1

        return self

    ####################################################################
    # PREDICT
    ####################################################################
    def predict(self, X):

        X = self._prepare_input(X)

        self.model.eval()

        with torch.no_grad():

            out = self.model(X)

        return out.argmax(dim=1).cpu().numpy()
    
    ####################################################################
    # OPTIONAL predict_proba
    ####################################################################

    def predict_proba(self, X):

        X = self._prepare_input(X)

        self.model.eval()

        with torch.no_grad():

            out = self.model(X)

            probs = torch.softmax(out, dim=1)

        return probs.cpu().numpy()