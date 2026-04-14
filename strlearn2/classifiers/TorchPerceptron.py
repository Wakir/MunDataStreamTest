import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # softmax w lossie

class TorchPerceptron:
    def __init__(self, input_dim, hidden_dim=128, lr=0.01):
        self.model = SimpleMLP(input_dim, hidden_dim)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.is_initialized = False

    def partial_fit(self, X, y, sample_weight=None):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        logits = self.model(X)
        losses = self.loss_fn(logits, y)

        if sample_weight is not None:
            w = torch.tensor(sample_weight, dtype=torch.float32)
            losses = losses * w

        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(X), dim=1).numpy()

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self.model(X), dim=1).numpy()
