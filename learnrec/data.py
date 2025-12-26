import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

CRITEO_SAMPLE_URL = "https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/criteo_sample.txt"


def _download_sample(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, "criteo_sample.txt")
    if os.path.exists(target_path):
        return target_path

    # Place the temporary file inside the target directory so the final rename
    # does not cross filesystems (e.g., /tmp -> mounted workspace), which avoids
    # "Invalid cross-device link" errors.
    with tempfile.NamedTemporaryFile(delete=False, dir=data_dir) as tmp:
        tmp_path = tmp.name
    try:
        import urllib.request

        urllib.request.urlretrieve(CRITEO_SAMPLE_URL, tmp_path)
        shutil.move(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return target_path


def load_criteo_dataset(data_dir: str = "data", sample_num: int = 200000, test_size: float = 0.2):
    """Download and preprocess a Criteo-style sample dataset.

    The loader follows the common CTR preprocessing recipe: missing values are filled,
    dense features are min-max scaled, and sparse features are label-encoded.
    """

    data_file = _download_sample(data_dir)
    column_names = [
        "label",
        *[f"I{i}" for i in range(1, 14)],
        *[f"C{i}" for i in range(1, 27)],
    ]
    data = pd.read_csv(data_file, sep="\t", names=column_names)

    if sample_num and len(data) > sample_num:
        data = data.sample(sample_num, random_state=42).reset_index(drop=True)

    dense_features: List[str] = [f"I{i}" for i in range(1, 14)]
    sparse_features: List[str] = [f"C{i}" for i in range(1, 27)]

    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna("-1")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = scaler.fit_transform(data[dense_features])

    feature_sizes: Dict[str, int] = {}
    for feat in sparse_features:
        encoder = LabelEncoder()
        data[feat] = encoder.fit_transform(data[feat])
        feature_sizes[feat] = data[feat].nunique()

    train, test = train_test_split(data, test_size=test_size, random_state=2024)
    train_y = torch.tensor(train["label"].values, dtype=torch.float32)
    test_y = torch.tensor(test["label"].values, dtype=torch.float32)

    train_dense = torch.tensor(train[dense_features].values, dtype=torch.float32)
    test_dense = torch.tensor(test[dense_features].values, dtype=torch.float32)

    train_sparse = torch.tensor(train[sparse_features].values, dtype=torch.long)
    test_sparse = torch.tensor(test[sparse_features].values, dtype=torch.long)

    history_sparse = [f"C{i}" for i in range(2, 7)]
    query_feature = "C1"
    max_history = len(history_sparse)

    def build_histories(df: pd.DataFrame):
        history = df[history_sparse].values
        lengths = torch.tensor([max_history] * len(df), dtype=torch.long)
        return torch.tensor(history, dtype=torch.long), lengths

    train_hist, train_len = build_histories(train)
    test_hist, test_len = build_histories(test)

    feature_info = {
        "dense_features": dense_features,
        "sparse_features": sparse_features,
        "feature_sizes": feature_sizes,
        "query_feature": query_feature,
        "history_features": history_sparse,
        "history_max_len": max_history,
    }

    train_dataset = CTRDataset(
        train_dense, train_sparse, train_hist, train_len, train_y, feature_info
    )
    test_dataset = CTRDataset(
        test_dense, test_sparse, test_hist, test_len, test_y, feature_info
    )
    return train_dataset, test_dataset, feature_info


class CTRDataset(torch.utils.data.Dataset):
    """A simple PyTorch dataset for CTR experiments."""

    def __init__(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
        history: torch.Tensor,
        history_length: torch.Tensor,
        labels: torch.Tensor,
        feature_info: Dict,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.history = history
        self.history_length = history_length
        self.labels = labels
        self.feature_info = feature_info

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "dense": self.dense[idx],
            "sparse": self.sparse[idx],
            "history": self.history[idx],
            "history_length": self.history_length[idx],
            "label": self.labels[idx],
        }


def build_dataloaders(train_ds, test_ds, batch_size: int = 512) -> Tuple:
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader
