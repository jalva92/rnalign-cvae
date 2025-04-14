# Installation

Follow these steps to set up the cVAE gene expression repository.

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cvae-gene-expression.git
cd cvae-gene-expression
```

## 2. Create a Python Environment

It is recommended to use Python 3.8 or later.

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

## 4. (Optional) Install in Editable Mode

To use the `cvae` package in development:

```bash
pip install -e .
```

## 5. Verify Installation

You can check that the package and scripts are available:

```bash
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/hyperparameter_tuning.py --help
```

## Requirements

- Python 3.8+
- torch
- pyro-ppl
- numpy
- pandas
- scikit-learn
- pyyaml

All dependencies are listed in `requirements.txt`.