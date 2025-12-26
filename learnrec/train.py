import argparse
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .data import build_dataloaders, load_criteo_dataset
from .models import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        dense = batch["dense"].to(device)
        sparse = batch["sparse"].to(device)
        history = batch["history"].to(device)
        history_length = batch["history_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        if isinstance(model, (nn.Module,)):
            if model.__class__.__name__ == "MatrixFactorization":
                logits = model(sparse)
            elif model.__class__.__name__ in {"FactorizationMachine", "FieldAwareFactorizationMachine"}:
                logits = model(sparse)
            elif model.__class__.__name__ == "WideAndDeep":
                logits = model(dense, sparse)
            elif model.__class__.__name__ == "DeepCrossNetwork":
                logits = model(dense, sparse)
            elif model.__class__.__name__ == "DeepInterestNetwork":
                logits = model(dense, sparse, history, history_length)
            else:
                raise ValueError(f"Unsupported model {model.__class__.__name__}")
        else:
            raise ValueError("Model must be a torch.nn.Module")

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            dense = batch["dense"].to(device)
            sparse = batch["sparse"].to(device)
            history = batch["history"].to(device)
            history_length = batch["history_length"].to(device)
            label = batch["label"].to(device)

            if model.__class__.__name__ == "MatrixFactorization":
                logits = model(sparse)
            elif model.__class__.__name__ in {"FactorizationMachine", "FieldAwareFactorizationMachine"}:
                logits = model(sparse)
            elif model.__class__.__name__ == "WideAndDeep":
                logits = model(dense, sparse)
            elif model.__class__.__name__ == "DeepCrossNetwork":
                logits = model(dense, sparse)
            elif model.__class__.__name__ == "DeepInterestNetwork":
                logits = model(dense, sparse, history, history_length)
            else:
                raise ValueError(f"Unsupported model {model.__class__.__name__}")

            preds.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(label.cpu().numpy())
    auc = roc_auc_score(labels, preds)
    return auc


def main(args: Dict = None):
    parser = argparse.ArgumentParser(description="CTR model trainer")
    parser.add_argument("--model", choices=["mf", "fm", "ffm", "wdl", "dcn", "din"], default="fm")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample_num", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed_dim", type=int, default=8)
    parsed = parser.parse_args(args)

    device = torch.device(parsed.device if torch.cuda.is_available() or parsed.device == "cpu" else "cpu")
    train_ds, test_ds, feature_info = load_criteo_dataset(parsed.data_dir, sample_num=parsed.sample_num)
    train_loader, test_loader = build_dataloaders(train_ds, test_ds, batch_size=parsed.batch_size)

    model = build_model(parsed.model, feature_info, embed_dim=parsed.embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parsed.lr)

    for epoch in range(parsed.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        auc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, AUC={auc:.4f}")


if __name__ == "__main__":
    main()
