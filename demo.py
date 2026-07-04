#!/usr/bin/env python3
"""Full pipeline in simulation mode: dataset, training, sweep, evaluation."""

from __future__ import annotations

import json
import shutil
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from data.build_dataset import build
from finetune.evaluate import full_report
from finetune.train import expand_sweep, load_config, train_simulated


def main() -> None:
    shutil.rmtree("runs", ignore_errors=True)

    counts = build()
    print(f"Dataset: {counts}\n")

    config = load_config("configs/lora_r16.yaml")
    summary = train_simulated(config)
    print(f"Training ({summary['mode']}): best val loss"
          f" {summary['best_val_loss']} at step"
          f" {summary['best_checkpoint_step']};"
          f" adapter at {summary['adapter']}\n")

    print("Hyperparameter sweep (first 6 of the grid):")
    rows = []
    for sweep_config in expand_sweep("configs/sweep.yaml")[:6]:
        result = train_simulated(sweep_config)
        rows.append((sweep_config["lora"]["r"],
                     sweep_config["training"]["learning_rate"],
                     sweep_config["training"]["epochs"],
                     result["best_val_loss"]))
    print(f"{'rank':>5} {'lr':>8} {'epochs':>7} {'val_loss':>9}")
    for rank, lr, epochs, loss in sorted(rows, key=lambda r: r[3]):
        print(f"{rank:>5} {lr:>8} {epochs:>7} {loss:>9}")

    print("\nBase vs fine-tuned on the benchmark:")
    report = full_report()
    print(f"  accuracy: {report['base_accuracy']:.1%} ->"
          f" {report['tuned_accuracy']:.1%}"
          f" ({report['improvements']} improvements,"
          f" {report['regressions']} regressions)")
    print(f"  per-category (tuned): {report['per_category_tuned']}")
    print(f"  judge: base {report['judge']['judge_mean_base']} vs tuned"
          f" {report['judge']['judge_mean_tuned']}"
          f" ({report['judge']['tuned_wins']} wins,"
          f" {report['judge']['tuned_losses']} losses)")
    print(f"  forgetting check: base {report['forgetting']['base_general']:.0%}"
          f" vs tuned {report['forgetting']['tuned_general']:.0%}"
          " on general probes")

    with open("runs/experiment_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nExperiment report written to runs/experiment_report.json")


if __name__ == "__main__":
    main()
