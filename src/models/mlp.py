import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')   # Mac M-series GPU acceleration
    else:
        return torch.device('cpu')


class MetDataset(Dataset):
    def __init__(self, X_sparse, y):
        if hasattr(X_sparse, 'toarray'):
            X_dense = X_sparse.toarray()
        else:
            X_dense = np.array(X_sparse)
        self.X = torch.FloatTensor(X_dense)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MetMLP(nn.Module):
    def __init__(self, input_dim=215, hidden_dims=[512, 256, 128], num_classes=21, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    avg_loss = total_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds)
    return avg_loss, y_pred


def train(X_train, y_train, X_test, y_test, epochs=50, batch_size=512,
          lr=0.001, hidden_dims=[512, 256, 128], dropout=0.3):
    device = get_device()
    print(f"Using device: {device}")

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    train_dataset = MetDataset(X_train, y_train)
    test_dataset = MetDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset.X.shape[1]
    num_classes = len(classes)
    model = MetMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes, dropout=dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_pred = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, y_pred


def save(model, path='models/mlp_model.pt'):
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    torch.save(model.state_dict(), full_path)
    print(f"Saved model → {full_path}")


if __name__ == "__main__":
    import scipy.sparse as sp

    from src.models.evaluate import load_data, plot_confusion_matrix, print_metrics, save_report

    parser = argparse.ArgumentParser(description="Train MLP on Met Museum data.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dims", type=str, default="512,256,128")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    data = load_data()
    X_train_csr = sp.csr_matrix(data["X_train"])
    X_test_csr = sp.csr_matrix(data["X_test"])

    model, y_pred = train(
        X_train_csr, data["y_train"],
        X_test_csr, data["y_test"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )

    report = print_metrics(data["y_test"], y_pred, data["le"].classes_, "mlp")
    save_report(report, "mlp")
    plot_confusion_matrix(data["y_test"], y_pred, data["le"].classes_, "mlp")
    save(model)
