import copy
import warnings
from typing import List, Tuple

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

FEATURES = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
DATASET_NAME = "hitorilabs/iris"


def build_federated_dataset(num_clients: int) -> FederatedDataset:
    partitioner = IidPartitioner(num_partitions=num_clients)
    return FederatedDataset(
        dataset=DATASET_NAME,
        partitioners={"train": partitioner},
    )


def load_seed_data(num_clients: int) -> Tuple[np.ndarray, np.ndarray]:
    fds = build_federated_dataset(num_clients)
    full_df = fds.load_split("train").with_format("pandas")[:]
    seed_rows = full_df.groupby("species", sort=False).head(1)
    X_seed = seed_rows[FEATURES].to_numpy(dtype=np.float64)
    y_seed = seed_rows["species"].to_numpy()
    return X_seed, y_seed


def load_client_partition(cid: int, num_clients: int):
    fds = build_federated_dataset(num_clients)
    df = fds.load_partition(cid, "train").with_format("pandas")[:]
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["species"].to_numpy()

    split_idx = int(0.8 * len(X))
    if split_idx <= 0:
        split_idx = 1
    if split_idx >= len(X):
        split_idx = len(X) - 1

    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def load_all_test_sets(num_clients: int):
    X_tests = []
    y_tests = []
    for cid in range(num_clients):
        _, _, X_test, y_test = load_client_partition(cid, num_clients)
        X_tests.append(X_test)
        y_tests.append(y_test)
    return np.concatenate(X_tests, axis=0), np.concatenate(y_tests, axis=0)


def create_model_template(num_clients: int) -> LogisticRegression:
    X_seed, y_seed = load_seed_data(num_clients)
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,
        warm_start=True,
        solver="saga",
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_seed, y_seed)

    model.coef_ = np.zeros_like(model.coef_, dtype=np.float64)
    model.intercept_ = np.zeros_like(model.intercept_, dtype=np.float64)
    return model


def get_model_params(model: LogisticRegression) -> List[np.ndarray]:
    return [model.coef_.copy(), model.intercept_.copy()]


def set_model_params(model: LogisticRegression, params: List[np.ndarray]) -> None:
    model.coef_ = np.array(params[0], dtype=np.float64, copy=True)
    model.intercept_ = np.array(params[1], dtype=np.float64, copy=True)


def local_train(X_train, y_train, global_params, model_template, local_epochs: int = 1):
    model = copy.deepcopy(model_template)
    set_model_params(model, global_params)

    if len(np.unique(y_train)) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(local_epochs):
                model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_proba, labels=model.classes_)
    train_acc = model.score(X_train, y_train)

    return {
        "params": get_model_params(model),
        "num_examples": len(X_train),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
    }


def aggregate_fedavg(updates):
    total_examples = sum(update["num_examples"] for update in updates)
    aggregated = [
        np.zeros_like(updates[0]["params"][0], dtype=np.float64),
        np.zeros_like(updates[0]["params"][1], dtype=np.float64),
    ]

    for update in updates:
        weight = update["num_examples"] / total_examples
        aggregated[0] += weight * update["params"][0]
        aggregated[1] += weight * update["params"][1]

    return aggregated


def evaluate_global_model(global_params, model_template, X_test, y_test):
    model = copy.deepcopy(model_template)
    set_model_params(model, global_params)
    proba = model.predict_proba(X_test)
    loss = log_loss(y_test, proba, labels=model.classes_)
    acc = model.score(X_test, y_test)
    return float(loss), float(acc)
