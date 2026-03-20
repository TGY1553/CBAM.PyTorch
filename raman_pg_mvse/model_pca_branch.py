from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for model_pca_branch.py") from exc


@dataclass
class TorchTrainingResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    val_loss: float


class BranchedPCAMLP(nn.Module):
    def __init__(self, input_dims: Tuple[int, int, int], num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.branch_high = self._make_branch(input_dims[0], dropout)
        self.branch_mid = self._make_branch(input_dims[1], dropout)
        self.branch_low = self._make_branch(input_dims[2], dropout)
        fusion_dim = 32 * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _make_branch(input_dim: int, dropout: float) -> nn.Module:
        hidden = max(8, min(32, input_dim * 2))
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.ReLU(),
        )

    def forward(self, x_high, x_mid, x_low):
        h1 = self.branch_high(x_high)
        h2 = self.branch_mid(x_mid)
        h3 = self.branch_low(x_low)
        fusion = torch.cat([h1, h2, h3], dim=1)
        return self.classifier(fusion)


class Shallow1DCNN(nn.Module):
    def __init__(self, input_length: int, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BranchedPCAMLPWrapper:
    def __init__(self, n_components: int = 30, epochs: int = 250, batch_size: int = 16, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.25, patience: int = 25, random_state: int = 42) -> None:
        self.n_components = n_components
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state
        self.pca_: Optional[PCA] = None
        self.model_: Optional[BranchedPCAMLP] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dims_: Optional[Tuple[int, int, int]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BranchedPCAMLPWrapper":
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        n_components = max(n_components, min(6, X.shape[1]))
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        scores = self.pca_.fit_transform(X)
        split = self._split_scores(scores)
        self.input_dims_ = tuple(part.shape[1] for part in split)

        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.2,
            stratify=y_encoded,
            random_state=self.random_state,
        )
        model = BranchedPCAMLP(self.input_dims_, len(self.label_encoder_.classes_), dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_loader = self._make_loader(scores[train_idx], y_encoded[train_idx], shuffle=True)
        val_loader = self._make_loader(scores[val_idx], y_encoded[val_idx], shuffle=False)

        best_model = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        patience_counter = 0
        for _ in range(self.epochs):
            model.train()
            for xb1, xb2, xb3, yb in train_loader:
                xb = [xb1.to(self.device_), xb2.to(self.device_), xb3.to(self.device_)]
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                logits = model(*xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_loss = self._eval_loss(model, val_loader, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        model.load_state_dict(best_model)
        self.model_ = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.pca_.transform(X)
        loader = self._make_loader(scores, None, shuffle=False)
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for xb in loader:
                tensors = [tensor.to(self.device_) for tensor in xb]
                logits = self.model_(*tensors)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                outputs.append(prob)
        return np.vstack(outputs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)
        indices = np.argmax(prob, axis=1)
        return self.label_encoder_.inverse_transform(indices)

    def _split_scores(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = scores.shape[1]
        cut1 = max(2, n // 3)
        cut2 = max(cut1 + 2, 2 * n // 3)
        return scores[:, :cut1], scores[:, cut1:cut2], scores[:, cut2:]

    def _make_loader(self, scores: np.ndarray, labels: Optional[np.ndarray], shuffle: bool) -> DataLoader:
        parts = self._split_scores(scores)
        tensors = [torch.tensor(part, dtype=torch.float32) for part in parts]
        if labels is None:
            dataset = TensorDataset(*tensors)
        else:
            dataset = TensorDataset(*tensors, torch.tensor(labels, dtype=torch.long))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _eval_loss(self, model, loader, criterion) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for xb1, xb2, xb3, yb in loader:
                xb = [xb1.to(self.device_), xb2.to(self.device_), xb3.to(self.device_)]
                yb = yb.to(self.device_)
                logits = model(*xb)
                losses.append(float(criterion(logits, yb).cpu().item()))
        return float(np.mean(losses)) if losses else float("inf")


class Shallow1DCNNWrapper:
    def __init__(self, epochs: int = 200, batch_size: int = 16, lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.25, patience: int = 20, random_state: int = 42) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.patience = patience
        self.random_state = random_state
        self.model_: Optional[Shallow1DCNN] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Shallow1DCNNWrapper":
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, stratify=y_encoded, random_state=self.random_state
        )
        model = Shallow1DCNN(X.shape[1], len(self.label_encoder_.classes_), dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        train_loader = self._make_loader(X[train_idx], y_encoded[train_idx], shuffle=True)
        val_loader = self._make_loader(X[val_idx], y_encoded[val_idx], shuffle=False)

        best_model = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        patience_counter = 0
        for _ in range(self.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            val_loss = self._eval_loss(model, val_loader, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        model.load_state_dict(best_model)
        self.model_ = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        loader = self._make_loader(X, None, shuffle=False)
        self.model_.eval()
        outputs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_)
                logits = self.model_(xb)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(outputs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)
        indices = np.argmax(prob, axis=1)
        return self.label_encoder_.inverse_transform(indices)

    def _make_loader(self, X: np.ndarray, y: Optional[np.ndarray], shuffle: bool) -> DataLoader:
        x_tensor = torch.tensor(X[:, None, :], dtype=torch.float32)
        if y is None:
            dataset = TensorDataset(x_tensor)
        else:
            dataset = TensorDataset(x_tensor, torch.tensor(y, dtype=torch.long))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _eval_loss(self, model, loader, criterion) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                logits = model(xb)
                losses.append(float(criterion(logits, yb).cpu().item()))
        return float(np.mean(losses)) if losses else float("inf")
