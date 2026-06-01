"""
H3: Post-Variational Strategies Outperform Variational Training on Non-Convex QML Landscapes
=============================================================================================

Hypothesis: Post-variational methods (observable construction, ansatz expansion,
hybrid) consistently achieve higher test accuracy than fully variational training
on non-convex benchmark tasks from the 'Better than classical?' study (arXiv:2403.07059).

References:
  - arXiv:2307.10560 — Post-variational quantum neural networks
  - arXiv:2403.07059 — Better than classical? QML benchmarking
  - PennyLane demo: tutorial_post-variational_quantum_neural_networks

Baseline: demonstrations_v2/tutorial_post-variational_quantum_neural_networks/demo.py
"""

import pennylane as qp
from pennylane import numpy as np
from itertools import combinations
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ---------------------------------------------------------------------------
# Quantum components (adapted from the PennyLane post-variational demo)
# ---------------------------------------------------------------------------
def feature_map(features, n_wires=4):
    for i in range(n_wires):
        qp.Hadamard(i)
    for i in range(len(features)):
        if i % 2:
            qp.AngleEmbedding(features=features[i], wires=range(n_wires), rotation="Z")
        else:
            qp.AngleEmbedding(features=features[i], wires=range(n_wires), rotation="X")


def ansatz(params, n_wires=4):
    for i in range(n_wires):
        qp.RY(params[i], wires=i)
    for i in range(n_wires):
        qp.CNOT(wires=[(i - 1) % n_wires, i % n_wires])
    for i in range(n_wires):
        qp.RY(params[i + n_wires], wires=i)
    for i in range(n_wires):
        qp.CNOT(wires=[(n_wires - 2 - i) % n_wires, (n_wires - i - 1) % n_wires])


def local_pauli_group(qubits, locality):
    return list(_gen_paulis(0, 0, "", qubits, locality))


def _gen_paulis(identities, paulis, output, qubits, locality):
    if len(output) == qubits:
        yield output
    else:
        yield from _gen_paulis(identities + 1, paulis, output + "I", qubits, locality)
        if paulis < locality:
            yield from _gen_paulis(identities, paulis + 1, output + "X", qubits, locality)
            yield from _gen_paulis(identities, paulis + 1, output + "Y", qubits, locality)
            yield from _gen_paulis(identities, paulis + 1, output + "Z", qubits, locality)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def variational_strategy(X_train, y_train, X_test, y_test, n_wires=4):
    dev = qp.device("default.qubit", wires=n_wires)

    @qp.qnode(dev)
    def circuit(params, features):
        feature_map(features, n_wires)
        ansatz(params, n_wires)
        return qp.expval(qp.PauliZ(0))

    weights = 0.01 * np.random.randn(2 * n_wires)
    bias = 0.0

    def variational_classifier(w, b, x):
        return circuit(w, x) + b

    def square_loss(labels, preds):
        return np.mean((labels - np.array(preds)) ** 2)

    lr = 0.1
    for epoch in range(30):
        for i in range(len(X_train)):
            pred = variational_classifier(weights, bias, X_train[i])
            loss = square_loss([y_train[i]], [pred])
            grad_weights = 2 * (pred - y_train[i]) * 1
            weights -= lr * grad_weights
            bias -= lr * grad_weights

    preds = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]
    return accuracy_score(y_test, preds)


def observable_construction(X_train, y_train, X_test, y_test, n_wires=4, locality=2):
    dev = qp.device("default.qubit", wires=n_wires)
    measurements = local_pauli_group(n_wires, locality)

    @qp.qnode(dev)
    def circuit(features):
        feature_map(features, n_wires)
        return [qp.expval(qp.pauli.string_to_pauli_word(m)) for m in measurements]

    new_X_train = np.array([circuit(x) for x in X_train])
    new_X_test = np.array([circuit(x) for x in X_test])

    clf = MLPClassifier(early_stopping=True, max_iter=200, random_state=42)
    clf.fit(new_X_train, y_train)
    return clf.score(new_X_test, y_test)


def main():
    print("=" * 72)
    print("H3: Post-Variational Strategies on Non-Convex QML Landscapes")
    print("=" * 72)

    # Benchmark datasets (from arXiv:2403.07059 style)
    datasets = {
        "moons_easy": make_moons(n_samples=200, noise=0.05, random_state=42),
        "moons_hard": make_moons(n_samples=200, noise=0.3, random_state=42),
        "blobs": make_classification(n_samples=200, n_features=4, n_informative=2,
                                      n_redundant=0, n_clusters_per_class=1, random_state=42),
    }

    strategies = {
        "Variational": variational_strategy,
        "Observable(1-local)": lambda X_tr, y_tr, X_te, y_te: observable_construction(
            X_tr, y_tr, X_te, y_te, n_wires=4, locality=1),
        "Observable(2-local)": lambda X_tr, y_tr, X_te, y_te: observable_construction(
            X_tr, y_tr, X_te, y_te, n_wires=4, locality=2),
        "MLP (classical)": lambda X_tr, y_tr, X_te, y_te: MLPClassifier(
            early_stopping=True, max_iter=200, random_state=42).fit(X_tr, y_tr).score(X_te, y_te),
    }

    results = {}
    for dset_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        y_train = (y_train - 0.5) * 2  # map to {-1, 1}
        y_test = (y_test - 0.5) * 2

        print(f"\n--- {dset_name} ---")
        results[dset_name] = {}
        for sname, sfn in strategies.items():
            acc = sfn(X_train, y_train, X_test, y_test)
            results[dset_name][sname] = float(acc)
            print(f"  {sname:>25}: {acc:.4f}")

    # Summary table
    print("\n" + "=" * 72)
    print("Summary: Test Accuracy")
    print("-" * 72)
    header = f"{'Dataset':<15}" + "".join(f"{s:<22}" for s in strategies)
    print(header)
    print("-" * 72)
    for dset_name in datasets:
        row = f"{dset_name:<15}"
        for sname in strategies:
            row += f"{results[dset_name][sname]:<22.4f}"
        print(row)

    # Check hypothesis: do post-variational methods beat variational on every dataset?
    print("\n--- Hypothesis check ---")
    for dset_name in datasets:
        var_acc = results[dset_name].get("Variational", 0)
        best_pv = max(v for k, v in results[dset_name].items() if k != "Variational" and k != "MLP (classical)")
        print(f"  {dset_name}: variational={var_acc:.4f}  best post-var={best_pv:.4f}  "
              f"PV beats Var={best_pv > var_acc}")


if __name__ == "__main__":
    main()
