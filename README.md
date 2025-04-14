# cVAE for Gene Expression Data

A clean, modular, and extensible Python package for training and evaluating conditional Variational Autoencoders (cVAE) on gene expression data, with support for continuous and categorical conditioning, interpretability, and reproducible research.

---

## Overview

This repository provides a robust framework for training conditional Variational Autoencoders (cVAE) on gene expression data, with a focus on biological interpretability and flexibility. The codebase is organized for clarity, modularity, and ease of use, following best practices for modern Python projects.

Key features include:
- **Modular design**: Core functionality, utilities, and scripts are cleanly separated.
- **Flexible data handling**: Easily adapt to new datasets and preprocessing pipelines.
- **Custom loss functions**: Support for regularized ELBO and distance correlation.
- **Interpretability**: SHAP-based tools for model explanation.
- **Reproducibility**: YAML-based configuration for experiments.
- **Extensive documentation and examples**.

---

## Folder Structure

```
cvae-gene-expression/
│
├── cvae/                  # Core Python package
│   ├── models/            # Model definitions (CVAE, custom layers)
│   ├── data/              # Data loaders and preprocessing
│   ├── training/          # Training loops and loss functions
│   └── utils/             # Metrics, interpretability, helpers
│
├── scripts/               # Command-line scripts (train, evaluate, tune)
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── configs/               # YAML configuration files for experiments
│   └── experiments/       # Experiment-specific configs
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation (API, usage, examples)
├── examples/              # Example scripts for common use cases
├── .gitignore
├── LICENSE
├── README.md
├── setup.py
└── requirements.txt
```

### Component Relationships

- **scripts/** use the core package (`cvae/`) for all training, evaluation, and tuning.
- **configs/** provide experiment parameters in YAML format, loaded by scripts.
- **notebooks/** and **examples/** demonstrate usage and analysis.
- **docs/** contains API and usage documentation.
- **tests/** ensure code quality and correctness.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cvae-gene-expression.git
   cd cvae-gene-expression
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install in editable mode:**
   ```bash
   pip install -e .
   ```

---

## Usage

### 1. Training a cVAE Model

You can train a model using the provided script and a YAML configuration file:

```bash
python scripts/train.py --config configs/experiments/experiment1.yaml
```

Or, override config parameters with command-line arguments:

```bash
python scripts/train.py --config configs/experiments/experiment1.yaml --num-epochs 500
```

### 2. Evaluating a Trained Model

```bash
python scripts/evaluate.py --config configs/experiments/experiment1.yaml --checkpoint path/to/model.pt
```

### 3. Hyperparameter Tuning

```bash
python scripts/hyperparameter_tuning.py --config configs/experiments/tuning.yaml
```

---

## Example Configuration (YAML)

```yaml
# configs/experiments/experiment1.yaml

seed: 123
cuda: true
fulloutput: true
expressionfile: data/expression.csv
purityfile: data/purity.csv
classfile: data/classes.csv
outputfolder: results/
num_epochs: 1000
sup_num: 900
val_num: 100
learning_rate: 0.001
batch_size: 50
z_dim: 200
hidden_layers: [2500, 1000]
annealingfactor: 0.5
gradregmultiplier: 0.1
corregmultiplier: 0.1
ysmultiplier: 1
decodersigma: 1.0
reparamepsilon: 1.0
adamweightdecay: 0.0001
dropoutp: 0.5
batchnorm: false
scaleys: false
shap: false
```

---

## Documentation

- **Usage Guides**: See `docs/usage.md`
- **Examples**: See `examples/` and `notebooks/`

---

## Citing

If you use this codebase in your research, please cite the repository and any relevant publications.

---

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.

---

## Contact

For questions or support, please open an issue or contact the maintainers.