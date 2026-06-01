"""
H5: Data-Reuploading with Trainable Scaling Outperforms Fixed Encoding on Small Benchmarks
============================================================================================

Hypothesis: L-layer data re-uploading with per-layer trainable scaling parameters
closes the accuracy gap to classical MLP classifiers on the 'Better than classical?'
benchmark suite (arXiv:2403.07059).

References:
  - arXiv:2403.07059 — Better than classical? QML benchmarking
  - arXiv:2104.00021 — Effect of data encoding on expressive power (Schuld et al., 2021)
  - PennyLane demo: tutorial_data_reuploading_classifier

Baseline: demonstrations_v2/tutorial_data_reuploading_classifier/demo.py
"""

import pennylane as qp
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ---------------------------------------------------------------------------
# Trainable scaling data-reuploading circuit
# ---------------------------------------------------------------------------
dev = qp.device("lightning.qubit", wires=1)


@qp.qnode(dev)
def reupload_circuit(params, x, y_label):
    """
    params: shape (n_layers, 3) — trainable weights
    x: shape (3,) — input padded to 3 dims (Rot needs 3 angles)
    y_label: density matrix of target label
    """
    for p in params:
        qp.Rot(*x, wires=0)
        qp.Rot(*p, wires=0)
    return qp.expval(qp.Hermitian(y_label, wires=[0]))


def density_matrix(state):
    return state * np.conj(state).T


label_0 = np.array([[1], [0]])
label_1 = np.array([[0], [1]])
state_labels = np.array([label_0, label_1], requires_grad=False)
dm_labels = [density_matrix(s) for s in state_labels]


def train_reupload(X_train, y_train, n_layers=3, epochs=10, lr=0.6):
    params = np.random.uniform(size=(n_layers, 3), requires_grad=True)
    opt = AdamOptimizer(lr, beta1=0.9, beta2=0.999)

    for epoch in range(epochs):
        for i in range(len(X_train)):
            dm = dm_labels[int(y_train[i])]
            params, _, _ = opt.step(
                lambda p, x, dm: (1 - reupload_circuit(p, x, dm)) ** 2,
                params, X_train[i], dm
            )
    return params


def test_reupload(params, X_test, y_test):
    preds = []
    for i in range(len(X_test)):
        fidelities = [reupload_circuit(params, X_test[i], dm) for dm in dm_labels]
        preds.append(int(np.argmax(fidelities)))
    return np.mean(np.array(preds) == y_test)


# ---------------------------------------------------------------------------
# Benchmark loader
# ---------------------------------------------------------------------------
def load_benchmark(name):
    if name == "moons":
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    elif name == "circles":
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    elif name == "blobs":
        X, y = make_classification(n_samples=200, n_features=4, n_informative=2,
                                    n_redundant=0, n_clusters_per_class=1, random_state=42)
    else:
        raise ValueError(f"Unknown benchmark: {name}")

    X = StandardScaler().fit_transform(X)
    # Pad to 3-dim for Rot gate
    if X.shape[1] < 3:
        X = np.pad(X, ((0, 0), (0, 3 - X.shape[1])), mode="constant")
    return X, y


def main():
    print("=" * 72)
    print("H5: Data-Reuploading with Trainable Scaling")
    print("=" * 72)

    benchmarks = ["moons", "circles", "blobs"]
    layer_counts = [1, 2, 3, 5]
    optimizers = ["Adam"]

    results = {}
    for bname in benchmarks:
        print(f"\n--- {bname} ---")
        X, y = load_benchmark(bname)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        results[bname] = {}
        for L in layer_counts:
            params = train_reupload(X_train, y_train, n_layers=L, epochs=8, lr=0.5)
            acc = test_reupload(params, X_test, y_test)
            results[bname][f"reupload(L={L})"] = float(acc)
            print(f"  L={L}: test_acc={acc:.4f}")

        # Classical MLP baseline
        mlp = MLPClassifier(early_stopping=True, max_iter=200, random_state=42)
        mlp.fit(X_train[:, :2] if X_train.shape[1] > 2 else X_train, y_train)
        mlp_acc = mlp.score(X_test[:, :2] if X_test.shape[1] > 2 else X_test, y_test)
        results[bname]["MLP"] = float(mlp_acc)
        print(f"  MLP: test_acc={mlp_acc:.4f}")

    # Summary
    print("\n" + "=" * 72)
    print("Summary: Test Accuracy")
    print("-" * 72)
    header = f"{'Benchmark':<12}"
    for L in layer_counts:
        header += f"{'L=' + str(L):<14}"
    header += f"{'MLP':<10}"
    print(header)
    print("-" * 72)
    for bname in benchmarks:
        row = f"{bname:<12}"
        for L in layer_counts:
            row += f"{results[bname].get(f'reupload(L={L})', 0):<14.4f}"
        row += f"{results[bname].get('MLP', 0):<10.4f}"
        print(row)

    # Hypothesis: can data-reuploading close the gap to MLP?
    print("\n--- Hypothesis check ---")
    for bname in benchmarks:
        best_q = max(v for k, v in results[bname].items() if k != "MLP")
        mlp_a = results[bname]["MLP"]
        gap = mlp_a - best_q
        print(f"  {bname}: best_reupload={best_q:.4f}  MLP={mlp_a:.4f}  gap={gap:.4f}")


if __name__ == "__main__":
    main()
