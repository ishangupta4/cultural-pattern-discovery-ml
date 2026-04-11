import argparse
import os
import time

import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on Met Museum data.")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "xgb", "mlp", "hierarchical"],
        required=True,
        help="Which model to train.",
    )
    # RF / XGB
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    # MLP
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dims", type=str, default="512,256,128",
                        help="Comma-separated hidden layer sizes, e.g. '512,256,128'.")
    parser.add_argument("--dropout", type=float, default=0.3)
    # Shared
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device string: 'cpu', 'cuda', 'mps', or 'auto' (MLP only).")
    args = parser.parse_args()

    from src.models.evaluate import load_data, plot_confusion_matrix, print_metrics, save_report

    data = load_data()
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    le      = data["le"]

    t0 = time.time()

    if args.model == "lr":
        from src.models.logistic_regression import save, train
        model  = train(X_train, y_train)
        y_pred = model.predict(X_test)
        report = print_metrics(y_test, y_pred, le.classes_, "lr")
        save_report(report, "lr")
        plot_confusion_matrix(y_test, y_pred, le.classes_, "lr")
        save(model)

    elif args.model == "rf":
        from src.models.random_forest import save, train
        model  = train(X_train, y_train, args.n_estimators, args.max_depth)
        y_pred = model.predict(X_test)
        report = print_metrics(y_test, y_pred, le.classes_, "rf")
        save_report(report, "rf")
        plot_confusion_matrix(y_test, y_pred, le.classes_, "rf")
        save(model)

    elif args.model == "xgb":
        from src.models.xgboost_model import save, train
        X_train_csr = sp.csr_matrix(X_train)
        X_test_csr  = sp.csr_matrix(X_test)
        model  = train(X_train_csr, y_train, args.n_estimators, args.max_depth,
                       args.learning_rate, args.device)
        y_pred = model.predict(X_test_csr)
        report = print_metrics(y_test, y_pred, le.classes_, "xgb")
        save_report(report, "xgb")
        plot_confusion_matrix(y_test, y_pred, le.classes_, "xgb")
        save(model)

    elif args.model == "mlp":
        from src.models.mlp import save, train
        hidden_dims = [int(d) for d in args.hidden_dims.split(",")]
        model, y_pred = train(
            X_train, y_train, X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
        )
        report = print_metrics(y_test, y_pred, le.classes_, "mlp")
        save_report(report, "mlp")
        plot_confusion_matrix(y_test, y_pred, le.classes_, "mlp")
        save(model)

    elif args.model == "hierarchical":
        from src.models.hierarchical import DEPARTMENT_GROUPS, predict, train
        from src.models.hierarchical import save as hier_save
        stage1_model, specialist_models, group_le, department_groups = train(
            X_train, y_train, le, DEPARTMENT_GROUPS, device=args.device,
        )
        y_pred = predict(X_test, stage1_model, specialist_models, group_le, le, DEPARTMENT_GROUPS)
        report = print_metrics(y_test, y_pred, le.classes_, "hierarchical")
        save_report(report, "hierarchical")
        plot_confusion_matrix(y_test, y_pred, le.classes_, "hierarchical")
        hier_save(stage1_model, specialist_models, group_le)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
