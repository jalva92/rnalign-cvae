# Usage Guide

This guide explains how to train, evaluate, and tune the cVAE model using the provided scripts and configuration system.

## 1. Training a Model

You can train a cVAE model using the `scripts/train.py` script. You can specify parameters via command-line arguments or a YAML configuration file.

### Using a YAML Config

```bash
python scripts/train.py --config configs/experiments/experiment1.yaml
```

### Using Command-Line Arguments

```bash
python scripts/train.py \
  --expressionfile data/expression.csv \
  --outputfolder outputs/run1 \
  --cohort mycohort \
  --z_dim 32 \
  --hidden_layers 128 128 \
  --num_epochs 100
```

See `python scripts/train.py --help` for all options.

## 2. Evaluating a Model

After training, evaluate your model using:

```bash
python scripts/evaluate.py \
  --model_file outputs/run1/results/my_model.save \
  --expressionfile data/expression.csv \
  --outputfolder outputs/run1/eval \
  --z_dim 32 \
  --hidden_layers 128 128
```

## 3. Hyperparameter Tuning

Run random search hyperparameter tuning:

```bash
python scripts/hyperparameter_tuning.py \
  --num_trials 10 \
  --output_dir tuning_results \
  --base_config configs/default_config.yaml
```

## 4. Configuration Files

- Default and experiment-specific configs are in `configs/`.
- You can override any config option via the command line.
- See `configs/default_config.yaml` for all available options and documentation.

## 5. Example Workflows

- See the `examples/` directory for example scripts.
- See the `notebooks/` directory for Jupyter notebooks demonstrating advanced usage and visualization.