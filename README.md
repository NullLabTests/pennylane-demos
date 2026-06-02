<p align="center">
  <a href="https://pennylane.ai/qml/demonstrations/" title="View PennyLane Demonstrations Online">
    <img src="/documentation/images/documentation-hero.png" width="100%" alt="PennyLane Demonstrations Hero Image">
  </a>
</p>

<h1 align="center">🔬 PennyLane Demonstrations — Agent Experiments Fork</h1>

<p align="center">
  <a href="https://github.com/PennyLaneAI/demos" title="Upstream Repository">
    <img src="https://img.shields.io/badge/Forked%20from-PennyLaneAI/demos-7C3AED?style=flat-square&logo=github" alt="Fork">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square&logo=apache" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python" alt="Python">
  </a>
  <a href="https://pennylane.ai">
    <img src="https://img.shields.io/badge/PennyLane-0.38+-teal.svg?style=flat-square&logo=quantum" alt="PennyLane">
  </a>
  <a href="https://github.com/NullLabTests/pennylane-agent-hypotheses">
    <img src="https://img.shields.io/badge/Hypotheses-north%20star-00D4AA?style=flat-square" alt="Hypotheses">
  </a>
  <a href="https://github.com/NullLabTests/pennylane-agent-experiments">
    <img src="https://img.shields.io/badge/Experiments-runnable-10B981?style=flat-square" alt="Experiments">
  </a>
  <a href="https://github.com/NullLabTests/pennylane-demos/pulse">
    <img src="https://img.shields.io/github/commit-activity/w/NullLabTests/pennylane-demos?style=flat-square&logo=git" alt="Activity">
  </a>
  <a href="https://github.com/NullLabTests/pennylane-demos/blob/master/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/Contributions-Welcome-orange?style=flat-square&logo=contributorcovenant" alt="Contributions">
  </a>
</p>

<p align="center">
  <b>Agent-driven experiments fork</b> of the PennyLane demonstration suite.<br/>
  Contains <b>5 Python implementations</b> of testable hypotheses in hybrid quantum-classical ML.<br/>
  <sub>⬆️ Inherits the full upstream demo collection from <a href="https://github.com/PennyLaneAI/demos">PennyLaneAI/demos</a>.</sub>
</p>

<br/>

<p align="center">
  <img src="documentation/images/agent_fork_overview.svg" alt="Fork Architecture Diagram" width="90%">
</p>
<p align="center"><sub>Fork architecture showing upstream origin, added experiments, and mapping of each hypothesis to its PennyLane baseline demo. Each experiment extends a specific tutorial.</sub></p>

<br/>

---

## 🌐 Ecosystem Overview

```mermaid
graph TB
    classDef north fill:#1a1a2e,stroke:#00d4aa,stroke-width:2,color:#ccc
    classDef impl fill:#1a1a2e,stroke:#7C3AED,stroke-width:3,color:#fff
    classDef exec fill:#1a1a2e,stroke:#10B981,stroke-width:2,color:#ccc
    classDef up fill:#333,stroke:#666,stroke-width:1,color:#999

    subgraph Upstream_Source["☁️ Upstream"]
        U[("🔬 PennyLaneAI/demos<br/>Official Demos")]
    end
    subgraph Agent_Stack["🤖 Agent Experiment Stack"]
        direction TB
        H[("🧪 agent-hypotheses<br/>Research Definitions")]
        D[("🐍 pennylane-demos<br/><b>← You Are Here</b>")]
        E[("📓 agent-experiments<br/>Runnable Notebooks")]
    end
    subgraph Deliverables["📦 Deliverables"]
        S[("📁 experiments/<br/>H1–H5 Python scripts")]
        R[("📊 results/<br/>Metrics & Plots")]
    end

    U -.->|"fork + extend"| D
    H -->|"defines"| D
    D -->|"implements"| S
    S -->|"packaged as"| E
    S --> R
    E -->|"results validate"| H

    class H north
    class D,impl,S,R impl
    class E exec
    class U up
```

<br/>

---

## 🧪 What's New: Agent Experiments (`experiments/`)

This fork adds **5 agent-generated experiment scripts** on top of the full PennyLane demo suite.

### Experiment Matrix

```mermaid
graph LR
    subgraph H1_H2["Trainability Track"]
        H1["H1: NAS HQNN"]
        H2["H2: Dissipation vs Local Cost"]
    end
    subgraph H3_H5["Expressivity Track"]
        H3["H3: Post-Variational"]
        H4["H4: PDE-Constrained"]
        H5["H5: Data-Reuploading"]
    end
    H1_H2 -->|"BP mitigation"| H3_H5
    style H1 fill:#8B5CF6,color:#fff
    style H2 fill:#3B82F6,color:#fff
    style H3 fill:#EF4444,color:#fff
    style H4 fill:#F59E0B,color:#fff
    style H5 fill:#10B981,color:#fff
```

| ID | Script | Area | Est. Cost | Dependencies | Run Command |
|:--:|--------|:----:|:---------:|:-----------|:------------|
| **H1** | [`h1_joint_nas_hqnn.py`](experiments/h1_joint_nas_hqnn.py) | <img src="https://img.shields.io/badge/Architecture%20Search-8B5CF6?style=flat-square"> | <img src="https://img.shields.io/badge/Medium-F59E0B?style=flat-square"> | `pennylane torch scikit-learn` | `python experiments/h1_joint_nas_hqnn.py` |
| **H2** | [`h2_engineered_dissipation_bp.py`](experiments/h2_engineered_dissipation_bp.py) | <img src="https://img.shields.io/badge/Trainability-3B82F6?style=flat-square"> | <img src="https://img.shields.io/badge/Medium-F59E0B?style=flat-square"> | `pennylane matplotlib` | `python experiments/h2_engineered_dissipation_bp.py` |
| **H3** | [`h3_post_variational_benchmarks.py`](experiments/h3_post_variational_benchmarks.py) | <img src="https://img.shields.io/badge/Expressivity-EF4444?style=flat-square"> | <img src="https://img.shields.io/badge/Low--Medium-84CC16?style=flat-square"> | `pennylane scikit-learn` | `python experiments/h3_post_variational_benchmarks.py` |
| **H4** | [`h4_pde_constrained_loss.py`](experiments/h4_pde_constrained_loss.py) | <img src="https://img.shields.io/badge/Trainability-3B82F6?style=flat-square"> | <img src="https://img.shields.io/badge/Medium-F59E0B?style=flat-square"> | `pennylane matplotlib` | `python experiments/h4_pde_constrained_loss.py` |
| **H5** | [`h5_data_reuploading_scaling.py`](experiments/h5_data_reuploading_scaling.py) | <img src="https://img.shields.io/badge/Expressivity-EF4444?style=flat-square"> | <img src="https://img.shields.io/badge/Low--Medium-84CC16?style=flat-square"> | `pennylane scikit-learn` | `python experiments/h5_data_reuploading_scaling.py` |

Each script is **self-contained** with CLI argument parsing (`--help` for options) and timestamped JSON output saved to `experiments/results/`.

<br/>

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# PennyLane (core + lightning simulator)
pip install pennylane pennylane-lightning
```

### Run All Experiments

```bash
# Clone this fork
git clone https://github.com/NullLabTests/pennylane-demos.git
cd pennylane-demos

# Run experiments sequentially
for script in experiments/h*.py; do
    echo "=== Running $script ==="
    python "$script"
    echo ""
done
```

### Run with Custom Parameters

Each script supports CLI arguments:

```bash
python experiments/h1_joint_nas_hqnn.py --n-samples 200 --n-epochs 20 --output-dir experiments/results
python experiments/h2_engineered_dissipation_bp.py           # uses defaults
python experiments/h3_post_variational_benchmarks.py        # uses defaults
python experiments/h4_pde_constrained_loss.py               # uses defaults
python experiments/h5_data_reuploading_scaling.py           # uses defaults
```

Results are saved to `experiments/results/<id>_<timestamp>.json` with plots in PNG format.

<br/>

---

## 📁 Repository Structure

```
📦 pennylane-demos (fork)
│
├── 🧪 AGENT EXPERIMENTS — Added content
│   ├── 📁 experiments/
│   │   ├── h1_joint_nas_hqnn.py              # H1: NAS for Pareto-optimal HQNNs
│   │   ├── h2_engineered_dissipation_bp.py    # H2: Engineered dissipation vs local cost
│   │   ├── h3_post_variational_benchmarks.py  # H3: Post-variational strategies
│   │   ├── h4_pde_constrained_loss.py         # H4: PDE-constrained loss functions
│   │   ├── h5_data_reuploading_scaling.py     # H5: Trainable data reuploading
│   │   └── 📁 results/                        # 📊 Generated metrics & plots (gitignored)
│   └── 📄 README.md                           # ℹ️ This file
│
├── ⬆️ UPSTREAM CONTENT — Inherited from PennyLaneAI/demos
│   ├── 📁 demonstrations_v2/                  # 🌐 Official PennyLane demo notebooks
│   ├── 📁 dependencies/                       # 📦 Dependency metadata
│   ├── 📁 documentation/                      # 📖 CLI tool & docs
│   ├── 📁 lib/                                # 🛠️ Shared utilities
│   ├── 📁 _static/                            # 🎨 Web assets
│   ├── 📁 _templates/                         # 📝 Sphinx templates
│   ├── 📄 conf.py                             # ⚙️ Sphinx configuration
│   ├── 📄 index.rst                           # 📑 Documentation index
│   ├── 📄 CONTRIBUTING.md                     # 📖 Contributor guide
│   └── 📄 LICENSE                             # ⚖️ Apache 2.0
│
└── 🛠️ BUILD & CI
    ├── 📄 pyproject.toml                      # 🐍 Poetry project config
    ├── 📄 poetry.lock                         # 🔒 Locked dependencies
    ├── 📄 extension.py                        # 🔧 Sphinx extension
    └── 📁 .github/                            # 🤖 CI workflows
```

<br/>

---

## 🔗 Sister Repositories

| Repository | Badge | Purpose | Link |
|-----------|-------|---------|------|
| **agent-hypotheses** | <img src="https://img.shields.io/badge/Purpose-Definition-00D4AA?style=flat-square"> | Research hypotheses & metrics | [→ View](https://github.com/NullLabTests/pennylane-agent-hypotheses) |
| **pennylane-demos** (this repo) | <img src="https://img.shields.io/badge/Purpose-Implementation-7C3AED?style=flat-square"> | Python scripts (forked) | ← You are here |
| **agent-experiments** | <img src="https://img.shields.io/badge/Purpose-Execution-10B981?style=flat-square"> | Notebooks + CI | [→ View](https://github.com/NullLabTests/pennylane-agent-experiments) |

### Data Flow

```mermaid
flowchart LR
    subgraph Up[" "]
        U["PennyLaneAI/demos"] -.->|"basis for"| D["pennylane-demos"]
    end
    subgraph Core[" "]
        H["agent-hypotheses"] -->|"informs"| D
    end
    subgraph Down[" "]
        D -->|"wraps into"| E["agent-experiments"]
    end
    style U fill:#333,stroke:#666,color:#999,stroke-width:1
    style H fill:#1a1a2e,stroke:#00d4aa,color:#fff,stroke-width:2
    style D fill:#1a1a2e,stroke:#7C3AED,color:#fff,stroke-width:3
    style E fill:#1a1a2e,stroke:#10B981,color:#fff,stroke-width:2
```

<br/>

<p align="center">
  <img src="documentation/images/data_flow.svg" alt="Cross-Repository Data Flow" width="90%">
</p>
<p align="center"><sub>End-to-end data flow: hypotheses.json defines experiment parameters → Python scripts execute with CLI args → timestamped JSON results validate hypotheses. A complete feedback loop across all three repositories.</sub></p>

<br/>

---

## ⬆️ Upstream Attribution

This fork inherits the full [PennyLane demonstration suite](https://pennylane.ai/qml/demonstrations) — a collection of tutorials and implementations ranging from introductory concepts to cutting-edge quantum computing research. Built by [Xanadu](https://xanadu.ai/), for research.

| Resource | Link |
|----------|------|
| 🌐 Online Demonstrations | [pennylane.ai/qml/demonstrations](https://pennylane.ai/qml/demonstrations) |
| 📖 Contributing Guide | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 🛠️ CLI Tool | [`documentation/demo-cli.md`](/documentation/demo-cli.md) |
| 📦 Dependency Management | [`dependencies/README.md`](/dependencies/README.md) |
| 🐛 Issue Tracker | [PennyLaneAI/demos/issues](https://github.com/PennyLaneAI/demos/issues) |

<br/>

---

## 📄 License

The materials and demonstrations contained within this repository are **free** and **open-source**, released under the **Apache License, Version 2.0**.

Please note, the file `custom_directives.py` is available under the **BSD 3-Clause License**, with copyright © 2017, PyTorch contributors.

---

<p align="center">
  <sub>Part of the <a href="https://github.com/NullLabTests">NullLabTests</a> agent-driven research ecosystem.</sub>
  <br/>
  <sub>
    <img src="https://img.shields.io/badge/-hypotheses-00D4AA?style=flat-square">
    <img src="https://img.shields.io/badge/-demos-7C3AED?style=flat-square">
    <img src="https://img.shields.io/badge/-experiments-10B981?style=flat-square">
  </sub>
</p>
