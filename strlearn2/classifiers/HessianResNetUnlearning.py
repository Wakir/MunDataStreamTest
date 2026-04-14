import numpy as np
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet18
from sklearn.base import BaseEstimator, ClassifierMixin


class HessianResNetUnlearning(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        window_size=5,
        unlearning_rate=1.0,
        lr=0.01,
        cg_iters=10,
        damping=0.01,
        epochs = 10,
        device=None,
    ):

        self.window_size = window_size
        self.unlearning_rate = unlearning_rate
        self.lr = lr
        self.cg_iters = cg_iters
        self.damping = damping
        self.epochs = epochs

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = resnet18(num_classes=10).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr
        )

        self.buffer = deque(maxlen=window_size)

        self.k = 0

        self.classes_ = None

        self.train_times_ = []
        self.memory_usage_ = []
    
    ####################################################################
    # MEMORY USAGE
    ####################################################################

    def _get_memory_usage(self):

        memory = 0

        # pamięć parametrów modelu
        for p in self.model.parameters():
            memory += p.nelement() * p.element_size()

        # pamięć gradientów (jeśli istnieją)
        for p in self.model.parameters():
            if p.grad is not None:
                memory += p.grad.nelement() * p.grad.element_size()

        # GPU memory (jeśli używany CUDA)
        if torch.cuda.is_available():
            memory += torch.cuda.memory_allocated(self.device)

        return memory


    ####################################################################
    # Utility functions
    ####################################################################

    def _prepare_input(self, X):

        if isinstance(X, np.ndarray):

            # NHWC -> NCHW
            if X.ndim == 4:
                X = np.transpose(X, (0, 3, 1, 2))

            # MNIST: 1 kanał -> 3 kanały
            if X.shape[1] == 1:
                X = np.repeat(X, 3, axis=1)

            # gdyby przyszło jako spłaszczone 784
            if X.ndim == 2 and X.shape[1] == 784:
                X = X.reshape(-1, 1, 28, 28)
                X = np.repeat(X, 3, axis=1)

        X = torch.tensor(X).float().to(self.device)

        return X


    def _prepare_target(self, y):

        return torch.tensor(y).long().to(self.device)


    def _params_to_vector(self):

        return torch.cat(
            [p.reshape(-1) for p in self.model.parameters()]
        )


    ####################################################################
    # TRAIN STEP
    ####################################################################

    def train_chunk(self, X, y, epochs):

        X = self._prepare_input(X)
        y = self._prepare_target(y)

        for _ in range(epochs):

            self.optimizer.zero_grad()

            out = self.model(X)

            loss = self.loss_fn(out, y)

            loss.backward()

            self.optimizer.step()


    ####################################################################
    # K1 Mean gradient
    ####################################################################

    def compute_mean_gradient(self, X, y):

        self.model.zero_grad()

        X = self._prepare_input(X)
        y = self._prepare_target(y)

        out = self.model(X)

        loss = self.loss_fn(out, y)

        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=True
        )

        grad_vec = torch.cat(
            [g.reshape(-1) for g in grads]
        )

        return grad_vec


    ####################################################################
    # K2 Hessian Vector Product
    ####################################################################

    def hessian_vector_product(self, loss, params, v):

        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True
        )

        grad_vec = torch.cat(
            [g.reshape(-1) for g in grads]
        )

        gv = torch.dot(grad_vec, v)

        Hv = torch.autograd.grad(
            gv,
            params,
            retain_graph=True
        )

        Hv_vec = torch.cat(
            [h.reshape(-1) for h in Hv]
        )

        Hv_vec += self.damping * v

        return Hv_vec


    ####################################################################
    # K3 Conjugate Gradient
    ####################################################################

    def conjugate_gradient(self, Av, b):

        x = torch.zeros_like(b)

        r = b.clone()

        p = r.clone()

        for _ in range(self.cg_iters):

            Ap = Av(p)

            alpha = torch.dot(r, r) / (torch.dot(p, Ap) + 1e-8)

            x = x + alpha * p

            r_new = r - alpha * Ap

            beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + 1e-8)

            p = r_new + beta * p

            r = r_new

        return x


    ####################################################################
    # K4 Apply parameter update
    ####################################################################

    def apply_update(self, delta):

        pointer = 0

        for param in self.model.parameters():

            numel = param.numel()

            upd = delta[pointer:pointer + numel].view(param.shape)

            param.data -= self.unlearning_rate * upd

            pointer += numel


    ####################################################################
    # UNLEARN STEP
    ####################################################################

    def unlearn_chunk(self, X, y):

        g = self.compute_mean_gradient(X, y)

        X = self._prepare_input(X)
        y = self._prepare_target(y)

        out = self.model(X)

        loss = self.loss_fn(out, y)

        params = list(self.model.parameters())

        def Av(v):
            return self.hessian_vector_product(loss, params, v)

        v = self.conjugate_gradient(Av, g)

        self.apply_update(v)


    ####################################################################
    # MAIN STREAM UPDATE
    ####################################################################

    def partial_fit(self, X, y, classes=None):

        print(self.k)
        t0 = time.perf_counter()

        if self.classes_ is None and classes is not None:
            self.classes_ = classes

        if self.k < self.window_size:

            self.train_chunk(X, y, self.epochs)

            self.buffer.append((X, y))

        else:

            X_old, y_old = self.buffer[0]

            self.unlearn_chunk(X_old, y_old)

            self.train_chunk(X, y, 1)

            self.buffer.append((X, y))

        self.train_times_.append(time.perf_counter() - t0)

        self.memory_usage_.append(self._get_memory_usage())

        self.k += 1

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