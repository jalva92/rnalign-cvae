# cVAE for Gene Expression Data

Code for RNAlign as presented as a [poster at ICLR 2025 AI4NA Workshop](https://drive.google.com/file/d/1cKzs5S_rHPV4bgFTH4m17e-c47Os1XCM/view?usp=sharing), for training and evaluating conditional Variational Autoencoders (cVAE) on gene expression data, with support for continuous and categorical conditioning, interpretability, and reproducible research.

This repository provides a robust framework for training conditional Variational Autoencoders (cVAE) on gene expression data, with a focus on biological interpretability and flexibility. 

---

## Usage

### This reposition is currently under development and will be updated to include instructions for training and validation.


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


## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
