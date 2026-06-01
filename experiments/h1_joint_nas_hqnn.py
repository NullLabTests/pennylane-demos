"""
H1: Joint Classical–Quantum NAS for Pareto-Optimal HQNN Architectures
=====================================================================

Hypothesis: Multi-objective neural architecture search (NAS) over a joint
classical–quantum design space discovers Pareto-optimal HQNN architectures
that simultaneously maximize accuracy and minimize computational cost (FLOPs).

Reference: arXiv:2605.25768, arXiv:2511.10062, arXiv:2402.10540

Baseline: demonstrations_v2/tutorial_variational_classifier/demo.py
"""

import pennylane as qp
from pennylane import numpy as np
import itertools
import time
import json

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "n_qubits": [2, 4, 6],
    "n_layers": [1, 2, 3, 4],
    "entanglement": ["linear", "circular", "sca"],
    "encoding": ["angle", "amplitude"],
}

def build_circuit(n_qubits, n_layers, entanglement, encoding):
    """Build a parameterized quantum circuit and return its QNode + param count."""
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit(inputs, weights):
        if encoding == "angle":
            qp.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        else:
            qp.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        qp.BasicEntanglerLayers(weights, wires=range(n_qubits), entanglement=entanglement)
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

    shape = (n_layers, n_qubits)
    n_params = n_layers * n_qubits
    return circuit, shape, n_params


def estimate_flops(n_qubits, n_layers, n_params):
    """Rough FLOPs proxy: each parameter ~1 forward pass with O(2^n_qubits) statevec ops."""
    return n_params * (2 ** n_qubits) * n_layers


# ---------------------------------------------------------------------------
# Benchmark: simple Moons classification (small subset for quick validation)
# ---------------------------------------------------------------------------
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(n_samples=100, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = StandardScaler().fit_transform(X)
    # Pad features to match qubit count if needed
    return X, y


def train_and_eval(circuit, weight_shape, X_train, y_train, X_test, y_test):
    """Train a circuit-based classifier with a classical output layer."""
    import torch
    torch.manual_seed(42)
    np.random.seed(42)

    n_qubits = weight_shape[1]
    qlayer = qp.qnn.TorchLayer(circuit, {"weights": weight_shape})

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], n_qubits),
        qlayer,
        torch.nn.Linear(n_qubits, 2),
        torch.nn.Softmax(dim=1),
    )

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    for epoch in range(10):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_te).argmax(dim=1).numpy()
    acc = (preds == y_test).mean()
    return acc


def main():
    print("=" * 72)
    print("H1: Joint Classical–Quantum NAS for Pareto-Optimal HQNNs")
    print("=" * 72)

    X, y = load_data(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = []
    keys = list(SEARCH_SPACE.keys())
    for values in itertools.product(*SEARCH_SPACE.values()):
        config = dict(zip(keys, values))
        n_qubits = config["n_qubits"]
        n_layers = config["n_layers"]

        if config["encoding"] == "amplitude" and X_train.shape[1] != 2 ** int(np.log2(n_qubits)):
            continue

        try:
            circuit, shape, n_params = build_circuit(
                n_qubits, n_layers, config["entanglement"], config["encoding"]
            )
            flops = estimate_flops(n_qubits, n_layers, n_params)
            t0 = time.time()
            acc = train_and_eval(circuit, shape, X_train, y_train, X_test, y_test)
            elapsed = time.time() - t0

            results.append({**config, "accuracy": float(acc), "flops": flops, "n_params": n_params,
                            "time_s": round(elapsed, 2)})
            print(f"nq={n_qubits} L={n_layers} ent={config['entanglement']:>8} "
                  f"enc={config['encoding']:>9} | acc={acc:.4f} flops={flops}")
        except Exception as e:
            print(f"SKIP {config}: {e}")

    # Pareto frontier (maximize accuracy, minimize FLOPs)
    sorted_res = sorted(results, key=lambda r: (-r["accuracy"], r["flops"]))
    pareto = []
    min_flops = float("inf")
    for r in sorted_res:
        if r["flops"] < min_flops:
            pareto.append(r)
            min_flops = r["flops"]

    print("\n--- Pareto-optimal architectures ---")
    for p in pareto:
        print(f"  acc={p['accuracy']:.4f} flops={p['flops']} qubits={p['n_qubits']} "
              f"layers={p['n_layers']} ent={p['entanglement']} enc={p['encoding']}")

    summary = {
        "hypothesis": "H1",
        "n_architectures_evaluated": len(results),
        "n_pareto_optimal": len(pareto),
        "pareto_front": pareto,
        "all_results": results,
    }
    with open("h1_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to h1_results.json ({len(results)} architectures)")


if __name__ == "__main__":
    main()
