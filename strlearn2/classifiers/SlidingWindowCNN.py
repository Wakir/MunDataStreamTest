# Usuń:
# from torchvision.models import resnet18

import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader
from collections import deque

from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================
# Prosty CNN dopasowujący się do kształtu wejścia
# ======================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()

        c, h, w = input_shape

        # Głębsza architektura + BatchNorm + Dropout
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(c, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),      # 28 -> 14
            nn.Dropout2d(0.10),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),      # 14 -> 7
            nn.Dropout2d(0.15),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Zamiast sztywnego 4x4
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

        # Inicjalizacja He (Kaiming)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity='relu'
                )
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================================================
# Sliding Window CNN
# ======================================================
class SlidingWindowCNN(BaseEstimator, ClassifierMixin):

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

        # Model zostanie zainicjalizowany po poznaniu kształtu danych
        self.model = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None

        self.buffer = deque(maxlen=window_size)
        self.k = 0

    # --------------------------------------------------
    # Inicjalizacja modelu na podstawie kształtu wejścia
    # --------------------------------------------------
    def _init_model(self, input_shape=(1, 28, 28), num_classes=10):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = SimpleCNN(
            input_shape=input_shape,
            num_classes=num_classes
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr
        )

        self._is_initialized = True

    # --------------------------------------------------
    # Przygotowanie danych wejściowych
    # --------------------------------------------------
    def _prepare_input(self, X):

        if isinstance(X, np.ndarray):

            # (N, 784) -> (N, 1, 28, 28)
            if X.ndim == 2 and X.shape[1] == 784:
                X = X.reshape(-1, 1, 28, 28)

            # (N, H, W) -> (N, 1, H, W)
            elif X.ndim == 3:
                X = X[:, None, :, :]

            # NHWC -> NCHW
            elif X.ndim == 4 and X.shape[-1] in (1, 3):
                X = np.transpose(X, (0, 3, 1, 2))

        X = torch.tensor(X).float().to(self.device)

        return X

    def _prepare_target(self, y):

        return torch.tensor(y).long().to(self.device)

    # --------------------------------------------------
    # Pobranie kształtu wejścia (C, H, W)
    # --------------------------------------------------
    def _infer_input_shape(self, X):

        if isinstance(X, np.ndarray):

            if X.ndim == 2 and X.shape[1] == 784:
                return (1, 28, 28)

            elif X.ndim == 3:
                # (N, H, W)
                return (1, X.shape[1], X.shape[2])

            elif X.ndim == 4:
                # NCHW
                if X.shape[1] in (1, 3):
                    return (X.shape[1], X.shape[2], X.shape[3])

                # NHWC
                elif X.shape[-1] in (1, 3):
                    return (X.shape[-1], X.shape[1], X.shape[2])

        raise ValueError(f"Nieobsługiwany kształt danych: {X.shape}")

    # --------------------------------------------------
    # Śledzenie użycia pamięci
    # --------------------------------------------------
    def _track_memory_usage(self):

        mem = 0

        if self.model is not None:
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
    # Trening na porcji danych
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

        self.model.train()

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

        # Inicjalizacja modelu po poznaniu kształtu danych
        if not self._is_initialized:
            input_shape = self._infer_input_shape(X)

            if classes is not None:
                num_classes = len(classes)
            elif self.classes_ is not None:
                num_classes = len(self.classes_)
            else:
                num_classes = int(np.max(y)) + 1

            self._init_model(
                input_shape=input_shape,
                num_classes=num_classes
            )

        # ==================================================
        # k < L → trening inkrementalny
        # ==================================================
        if self.k_ < self.window_size:

            self._train(X, y, self.epochs)
            self.buffer_.append((X, y))

        # ==================================================
        # k ≥ L → reset + trening na całym oknie
        # ==================================================
        else:
            self.buffer_.append((X, y))

            Xw = np.vstack([c[0] for c in self.buffer_])
            yw = np.hstack([c[1] for c in self.buffer_])

            input_shape = self._infer_input_shape(Xw)

            if self.classes_ is not None:
                num_classes = len(self.classes_)
            else:
                num_classes = int(np.max(yw)) + 1

            self._init_model(
                input_shape=input_shape,
                num_classes=num_classes
            )

            self._train(Xw, yw, self.epochs)

        t_end = time.perf_counter()

        self.train_times_.append(t_end - t_start)

        self._track_memory_usage()

        self.k_ += 1

        return self

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    def predict(self, X):

        X = self._prepare_input(X)

        self.model.eval()

        with torch.no_grad():
            out = self.model(X)

        return out.argmax(dim=1).cpu().numpy()

    # --------------------------------------------------
    # Predict_proba
    # --------------------------------------------------
    def predict_proba(self, X):

        X = self._prepare_input(X)

        self.model.eval()

        with torch.no_grad():
            out = self.model(X)
            probs = torch.softmax(out, dim=1)

        return probs.cpu().numpy()