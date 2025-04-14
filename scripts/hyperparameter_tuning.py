#!/usr/bin/env python
"""
Hyperparameter tuning script for the cVAE model.

This script generates random hyperparameter configurations, runs multiple training jobs,
and summarizes the results.
"""

import argparse
import os
import subprocess
import yaml
import numpy as np
import hashlib
import random

def seed_hash(*args):
    """Derive an integer hash from all args, for use as a random seed."""
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def random_hparams(random_seed, output={}):
    """Generate a random hyperparameter configuration."""
    def _hparam(name, random_val_fn):
        random_state = np.random.RandomState(seed_hash(random_seed, name))
        random_val = random_val_fn(random_state)
        output[name] = random_val

    _hparam('learning_rate', lambda r: float(10**r.uniform(-2, -5)))
    _hparam('batch_size', lambda r: int(r.choice([32, 64, 128, 256])))
    _hparam('num_layers', lambda r: int(r.choice([2, 3])))
    _hparam('z_dim', lambda r: int(2**r.uniform(7, 9)))
    _hparam('annealingfactor', lambda r: float(10**r.uniform(-1, 1.5)))
    _hparam('corregmultiplier', lambda r: float(10**r.uniform(-2, 0.5)))
    _hparam('gradregmultiplier', lambda r: float(10**r.uniform(-2, 0.5)))
    _hparam('ysmultiplier', lambda r: float(r.uniform(0, 100)))
    _hparam('decodersigma', lambda r: float(r.uniform(0.2, 1)))
    _hparam('reparamepsilon', lambda r: float(r.uniform(0.2, 1)))
    _hparam('adamweightdecay', lambda r: float(10**r.uniform(-6, -3)))
    _hparam('dropoutp', lambda r: float(r.choice([0.2, 0.4, 0.6])))
    return output

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for cVAE")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of random search trials")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store tuning results")
    parser.add_argument("--base_config", type=str, required=True, help="Base YAML config file for fixed parameters")
    parser.add_argument("--train_script", type=str, default="scripts/train.py", help="Path to train.py script")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.base_config, "r") as f:
        base_config = yaml.safe_load(f)

    results = []
    for trial in range(args.num_trials):
        trial_seed = random.randint(0, 1e6)
        hparams = random_hparams(trial_seed, output={})
        config = base_config.copy()
        config.update(hparams)
        config['seed'] = trial_seed
        config['hidden_layers'] = [128] * hparams['num_layers']  # Example: all layers 128 units

        config_path = os.path.join(args.output_dir, f"trial_{trial}_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        log_dir = os.path.join(args.output_dir, f"trial_{trial}_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"Running trial {trial+1}/{args.num_trials} with config: {config_path}")
        cmd = [
            "python", args.train_script,
            "--config", config_path
        ]
        # Optionally, add more CLI args if needed

        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        with open(os.path.join(log_dir, "stdout.txt"), "w") as f:
            f.write(result.stdout)
        with open(os.path.join(log_dir, "stderr.txt"), "w") as f:
            f.write(result.stderr)

        # Optionally, parse log files or output metrics for summary
        results.append({
            "trial": trial,
            "config": config_path,
            "returncode": result.returncode
        })

    # Save summary
    summary_path = os.path.join(args.output_dir, "tuning_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(results, f)

    print(f"Hyperparameter tuning complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()