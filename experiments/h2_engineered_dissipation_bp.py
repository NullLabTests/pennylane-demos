"""
H2: Engineered Dissipation Outperforms Local Cost Functions for BP Mitigation
==============================================================================

Hypothesis: Engineered Markovian dissipation added after each unitary layer
maintains higher gradient variance at large n and L compared to both global
and local cost baselines.

References:
  - npj Quantum Information 10, 88 (2024) — Engineered dissipation
  - Cerezo et al. (2021) — Local cost functions
  - arXiv:2411.08238 — NN-generated quantum states

Baselines: demonstrations_v2/tutorial_barren_plateaus/demo.py
           demonstrations_v2/tutorial_local_cost_functions/demo.py
"""

import pennylane as qp
from pennylane import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


np.random.seed(42)


def global_cost_circuit(params, n_qubits):
    """Global cost: expectation of |0><0|^⊗n."""
    for i in range(n_qubits):
        qp.RY(np.pi / 4, wires=i)
    for i in range(n_qubits):
        qp.RX(params[i], wires=i)
    for i in range(n_qubits - 1):
        qp.CZ(wires=[i, i + 1])
    H = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    H[0, 0] = 1
    return qp.expval(qp.Hermitian(H, list(range(n_qubits))))


def local_cost_circuit(params, n_qubits):
    """Local cost: average of single-qubit |0><0| terms."""
    for i in range(n_qubits):
        qp.RY(np.pi / 4, wires=i)
    for i in range(n_qubits):
        qp.RX(params[i], wires=i)
    for i in range(n_qubits - 1):
        qp.CZ(wires=[i, i + 1])
    # Local observable: single-qubit Pauli Z on first qubit
    return qp.expval(qp.PauliZ(0))


def dissipation_circuit(params, n_qubits):
    """Engineered dissipation via local amplitude damping on a mixed-state device."""
    for i in range(n_qubits):
        qp.RY(np.pi / 4, wires=i)
    for i in range(n_qubits):
        qp.RX(params[i], wires=i)
    for i in range(n_qubits - 1):
        qp.CZ(wires=[i, i + 1])
    # Engineered dissipation: apply weak amplitude damping to each qubit
    # Models Markovian energy relaxation (engineered loss channel)
    for i in range(n_qubits):
        qp.AmplitudeDamping(0.1, wires=i)
    return qp.expval(qp.PauliZ(0))


def compute_gradient_variance(circuit_fn, n_qubits, num_samples=80, is_dissipation=False):
    """Estimate gradient variance for a given circuit type."""
    if is_dissipation:
        dev = qp.device("default.mixed", wires=n_qubits)
    else:
        dev = qp.device("default.qubit", wires=n_qubits)
    qnode = qp.QNode(circuit_fn, dev, diff_method="parameter-shift")
    grad_fn = qp.grad(qnode)

    grads = []
    for _ in range(num_samples):
        params = np.random.uniform(-np.pi, np.pi, size=n_qubits)
        g = grad_fn(params, n_qubits=n_qubits)
        grads.append(g[-1])

    grads = np.array(grads)
    return float(np.var(grads)), float(np.mean(np.abs(grads)))


def main():
    print("=" * 72)
    print("H2: Engineered Dissipation vs Local Cost for BP Mitigation")
    print("=" * 72)

    qubit_range = [2, 3, 4, 5, 6, 8]
    strategies = {
        "Global cost": global_cost_circuit,
        "Local cost": local_cost_circuit,
        "Engineered dissipation": dissipation_circuit,
    }

    results = {}
    for name, fn in strategies.items():
        print(f"\n--- {name} ---")
        is_diss = name == "Engineered dissipation"
        var_list = []
        n_list = []
        for n in qubit_range:
            var, mean_abs = compute_gradient_variance(fn, n, num_samples=80, is_dissipation=is_diss)
            var_list.append(var)
            n_list.append(n)
            print(f"  n={n}: var={var:.6e}  mean|grad|={mean_abs:.6e}")
        results[name] = {"qubits": n_list, "variances": var_list}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    markers = {"Global cost": "o-", "Local cost": "s--", "Engineered dissipation": "D-."}
    for name, fn in strategies.items():
        d = results[name]
        axes[0].semilogy(d["qubits"], d["variances"], markers[name], label=name, linewidth=2)
    axes[0].set_xlabel("Number of qubits")
    axes[0].set_ylabel("Gradient variance")
    axes[0].set_title("Gradient variance vs qubit count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Fit exponential decay
    for name, fn in strategies.items():
        d = results[name]
        n_arr = np.array(d["qubits"])
        v_arr = np.array(d["variances"])
        coeffs = np.polyfit(n_arr[2:], np.log(v_arr[2:] + 1e-30), 1)
        axes[1].plot(n_arr[2:], np.exp(coeffs[1]) * np.exp(coeffs[0] * n_arr[2:]),
                     markers[name], label=f"{name} fit (slope={coeffs[0]:.2f})", alpha=0.6)
    axes[1].set_xlabel("Number of qubits")
    axes[1].set_ylabel("Gradient variance (fit)")
    axes[1].set_title("Exponential decay fit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("h2_barren_plateau_mitigation.png", dpi=150)
    print("\nFigure saved: h2_barren_plateau_mitigation.png")

    # Summary
    for name in strategies:
        d = results[name]
        v = d["variances"]
        print(f"\n{name:>25}: v(n=2)={v[0]:.2e}  v(n=8)={v[-1]:.2e}  "
              f"ratio={v[-1]/v[0]:.2e}")


if __name__ == "__main__":
    main()
