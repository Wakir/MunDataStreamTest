import numpy as np
import pandas as pd
import os
from tensorflow.keras.datasets import mnist, cifar10
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac, precision_score, recall_score
from specificity import specificity, specificity_macro
from strlearn2.classifiers import SlidingWindowPerceptron, HessianResNetUnlearning
from collections import defaultdict


def recovery_analysis(metric, rolling_metric, drift_chunk, max_chunk):

    if drift_chunk is None:
        return {
            "theta": None,
            "T_drop": None,
            "T_recovery": None,
            "recovery_time": None,
            "status": "no_drift"
        }

    drift_chunk_eval = drift_chunk - 1

    if drift_chunk_eval < 0 or drift_chunk_eval >= len(metric):
        return {
            "theta": None,
            "T_drop": None,
            "T_recovery": None,
            "recovery_time": None,
            "status": "drift_out_of_range"
        }

    post_drift = metric[drift_chunk_eval:]

    if len(post_drift) == 0:
        return {
            "theta": None,
            "T_drop": None,
            "T_recovery": None,
            "recovery_time": None,
            "status": "empty_post_drift"
        }

    # ================= NORMALNA ANALIZA =================

    min_val = min(post_drift)
    max_val = max(post_drift)
    theta = 0.90 * max_val

    # DROP
    T_drop = None
    for i in range(drift_chunk_eval, len(metric)):
        if metric[i] < 1.1 * min_val:
            T_drop = i
            break

    # RECOVERY
    T_recovery = None
    if T_drop is not None:
        for i in range(T_drop + 1, len(metric)):
            if metric[i] >= theta:
                T_recovery = i
                break

    recovery_time = None
    if T_drop is not None and T_recovery is not None:
        recovery_time = T_recovery - T_drop

    return {
        "theta": theta,
        "T_drop": T_drop,
        "T_recovery": T_recovery,
        "recovery_time": recovery_time,
        "status": "ok"
    }

class DataStream:
    def __init__(
        self,
        chunk_size: int,
        dataset_name: str,
        noise_percent: float,
        delta_noise: float,
        random_seed: int = 42,
    ):
        self.chunk_size = chunk_size
        self.dataset_name = dataset_name.upper()
        self.noise_percent = noise_percent
        self.delta_noise = delta_noise
        self.random_seed = random_seed

        self.rng = np.random.default_rng(random_seed)

        # Load data
        self.X, self.y = self._load_dataset()
        self.classes_ = np.unique(self.y)

        # Prepare chunks
        self._shuffle()
        self.chunks = self._create_balanced_chunks()

        # Required by strlearn
        self.n_chunks = len(self.chunks)
        self.reset()

        # Noise drift point
        self.noise_change_chunk = self.n_chunks // 2

    # --------------------------------------------------
    # Required API
    # --------------------------------------------------
    def reset(self):
        self.chunk_id = 0
        self.previous_chunk = None

    def __len__(self):
        return self.n_chunks

    def get_chunk(self, i=None):
        """
        Fully compatible with strlearn.Stream
        """
        # Sequential access
        if i is None:
            i = self.chunk_id

        if i >= self.n_chunks:
            raise IndexError("Chunk index out of range")

        idx = self.chunks[i]
        X_chunk = self.X[idx]
        y_chunk = self.y[idx]

        # Apply noise (concept drift)
        if i < self.noise_change_chunk:
            X_chunk = self._add_noise(X_chunk, self.noise_percent)
        else:
            X_chunk = self._add_noise(
                X_chunk,
                self.delta_noise
            )

        # Update internal state (CRUCIAL)
        self.previous_chunk = (X_chunk, y_chunk)
        self.chunk_id = i + 1

        return X_chunk, y_chunk

    # --------------------------------------------------
    # Dataset loading
    # --------------------------------------------------
    def _load_dataset(self):
        if self.dataset_name == "MNIST":
            (X1, y1), (X2, y2) = mnist.load_data()
            X = np.concatenate([X1, X2])
            y = np.concatenate([y1, y2])
            X = X[..., np.newaxis]

        elif self.dataset_name == "CIFAR-10":
            (X1, y1), (X2, y2) = cifar10.load_data()
            X = np.concatenate([X1, X2])
            y = np.concatenate([y1.flatten(), y2.flatten()])

        else:
            raise ValueError("Supported datasets: MNIST, CIFAR-10")

        return X.astype(np.float32) / 255.0, y

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _shuffle(self):
        idx = self.rng.permutation(len(self.X))
        self.X = self.X[idx]
        self.y = self.y[idx]

    def _create_balanced_chunks(self):
        per_class_indices = defaultdict(list)
        for i, label in enumerate(self.y):
            per_class_indices[label].append(i)

        n_classes = len(self.classes_)
        samples_per_class = self.chunk_size // n_classes

        pointers = {c: 0 for c in self.classes_}
        chunks = []

        while True:
            chunk_idx = []
            for c in self.classes_:
                start = pointers[c]
                end = start + samples_per_class
                if end > len(per_class_indices[c]):
                    return chunks
                chunk_idx.extend(per_class_indices[c][start:end])
                pointers[c] = end

            self.rng.shuffle(chunk_idx)
            chunks.append(chunk_idx)

    def _add_noise(self, X, noise_percent, sigma=0.5):
        if noise_percent <= 0:
            return X

        X_noisy = X.copy()
        N, H, W, C = X.shape
        total_pixels = H * W * C
        n_noisy = int(noise_percent * total_pixels)

        for i in range(N):
            idx = self.rng.choice(total_pixels, n_noisy, replace=False)
            noise = self.rng.normal(0, sigma, n_noisy)
            flat = X_noisy[i].reshape(-1)
            flat[idx] += noise
            X_noisy[i] = flat.reshape(H, W, C)

        return np.clip(X_noisy, 0.0, 1.0)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.chunk_id >= self.n_chunks:
            raise StopIteration
        return self.get_chunk()


    def is_dry(self):
        return self.chunk_id >= self.n_chunks - 1


def run_experiment(chunk_size, noise_percent, delta_noise, window_size, random_seed, ulr, learning_rate, metrics, alghoritm):
    stream = DataStream(
        chunk_size=chunk_size,
        dataset_name="MNIST",
        noise_percent=noise_percent,
        delta_noise=delta_noise,
        random_seed=random_seed,
    )

    if alghoritm=="Sliding":
        clf = SlidingWindowPerceptron(window_size=window_size, lr = learning_rate, epochs = ulr)
    elif alghoritm=="Unlearning":
        clf = HessianResNetUnlearning(window_size=window_size, unlearning_rate = ulr, lr=learning_rate)
    evaluator = TestThenTrain(metrics=list(metrics.values()))

    X0, y0 = next(iter(stream))
    clf.partial_fit(X0, y0, classes=stream.classes_)

    evaluator.process(stream, clf)

    scores = evaluator.scores[0]  # (metrics, time)
    train_times = np.array(clf.train_times_)
    memory = np.array(clf.memory_usage_)

    return {
        "metric_curves": {
            name: scores[:, i]
            for i, name in enumerate(metrics.keys())
        },
        "drift_chunk": stream.noise_change_chunk,
        "max_chunk": stream.n_chunks,
        "mean_time": train_times.mean(),
        "mean_memory": memory.mean(),
    }

import mlflow

def mlflow_run(chunk_size, noise_percent, delta_noise, window_size, random_seed, epochs, learning_rate, metrics):
    with mlflow.start_run(nested=True):

        mlflow.log_params({
            "chunk_size": chunk_size,
            "noise_percent": noise_percent,
            "delta_noise": delta_noise,
            "window_size": window_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
        })

        output = run_experiment(
            chunk_size,
            noise_percent,
            delta_noise,
            window_size,
            random_seed,
            epochs,
            learning_rate,
            metrics,
            "Sliding"
        )

        curves = output["metric_curves"]
        drift_chunk = output["drift_chunk"]
        max_chunk = output["max_chunk"]
        mean_time = output["mean_time"]
        mean_memory = output["mean_memory"]

        recovery_results_all = {}

        for metric_name, values in curves.items():

            # 📈 1. METRICS OVER TIME
            for step, value in enumerate(values):
                mlflow.log_metric(metric_name, float(value), step=step)

            # 📉 2. ROLLING
            rolling = (
                pd.Series(values)
                .rolling(window=5, min_periods=1)
                .mean()
                .values
            )

            # ♻️ 3. RECOVERY ANALYSIS
            recovery = recovery_analysis(
                values,
                rolling,
                drift_chunk,
                max_chunk
            )

            recovery_results_all[metric_name] = recovery

            # 🔹 4. SAVE TO MLFLOW
            if recovery["status"] == "ok":
                for k, v in recovery.items():
                    if k != "status" and v is not None:
                        mlflow.log_metric(f"{metric_name}_{k}", float(v))
            else:
                mlflow.log_param(f"{metric_name}_recovery_status", recovery["status"])
        mlflow.log_metric("drift_chunk", drift_chunk)
        mlflow.log_metric("mean_time", mean_time)
        mlflow.log_metric("mean_memory", mean_memory)

# HIPERPARAMETERS

chunk_sizes = [200]
noise_percents = [0.0]
new_noises = [0.0]
window_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
random_seeds = [42, 65, 88]
learning_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
epochs = [2, 4, 6, 8, 10]

from functools import partial

#NOT HIPERPARAMETERS (METRICS)
metrics = {
    "accuracy": accuracy_score,
    "balanced_accuracy": bac,
    "precision_macro": partial(precision_score, average="macro"),
    "recall_macro": partial(recall_score, average="macro"),
    "f1_macro": partial(f1_score, average="macro"),
    "specificity_macro": specificity_macro
}

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_experiment("MNIST_SuddenDrift")

import os
import json
import itertools

from joblib import Parallel, delayed
import itertools

# 🔽 GENEROWANIE TYLKO POPRAWNYCH KOMBINACJI
param_grid = [
    (chunk_size, noise_percent, delta_noise, window_size, epochs, learning_rates, random_seed,)
    for chunk_size, noise_percent, delta_noise, window_size, epochs, learning_rates, random_seed,
    in itertools.product(
        chunk_sizes,
        noise_percents,
        new_noises,
        window_sizes,
        epochs,
        learning_rates,
        random_seeds,
    )
    #if noise_percent != delta_noise
]

print(f"Liczba uruchamianych eksperymentów: {len(param_grid)}")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(mlflow_run)(
        chunk_size,
        noise_percent,
        delta_noise,
        window_size,
        random_seed,
        epochs,
        learning_rates,
        metrics,
    )
    for chunk_size, noise_percent, delta_noise, window_size, epochs, learning_rates, random_seed in param_grid
)

df = pd.DataFrame(results)
print(df.sort_values("accuracy", ascending=False).head())