# solver='saga' vs default 'lbfgs':
#   lbfgs stores a dense approximation of the Hessian — memory cost scales with
#   n_features^2, which is prohibitive on our ~200-feature sparse matrix at 388k rows.
#   SAGA is a stochastic gradient method that operates directly on sparse updates:
#   it never materializes a dense gradient for the full dataset, so memory stays
#   proportional to the batch size. It also converges faster than SGD on smooth
#   objectives because it uses variance reduction (stored past gradients).
#
# class_weight='balanced':
#   sklearn computes per-class weights as:
#     w_c = n_samples / (n_classes * n_samples_in_class_c)
#   These weights are applied to each sample's contribution to the loss, so a class
#   with 1/10 the samples gets 10x the loss weight — effectively up-weighting rare
#   departments (e.g. The Libraries, 534 records) relative to dominant ones
#   (e.g. Drawings & Prints, 172k records). This prevents the model from ignoring
#   minority classes to minimise overall cross-entropy.

import os

import joblib
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train(X_train, y_train):
    """Fit a balanced logistic regression on sparse training data. Returns fitted model."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save(model, path="models/lr_model.joblib"):
    """Save a fitted model to disk with joblib. Creates parent dirs if needed."""
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(model, full_path)
    print(f"Saved model → {full_path}")


if __name__ == "__main__":
    import time

    from src.models.evaluate import load_data, plot_confusion_matrix, print_metrics, save_report

    data = load_data()

    t0 = time.time()
    model = train(data["X_train"], data["y_train"])
    print(f"Training time: {time.time() - t0:.1f}s")

    y_pred = model.predict(data["X_test"])
    report = print_metrics(data["y_test"], y_pred, data["le"].classes_, "lr")
    save_report(report, "lr")
    plot_confusion_matrix(data["y_test"], y_pred, data["le"].classes_, "lr")
    save(model)
