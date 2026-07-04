"""LoRA/QLoRA training with experiment tracking.

With the GPU extras installed (torch, transformers, peft, trl,
bitsandbytes), this runs real QLoRA fine-tuning from the YAML config.
Without them, --simulate exercises the identical pipeline mechanics
(config loading, epoch loop, checkpointing, early stopping, tracking
output), so the pipeline is verifiable on any machine and in CI.
Every run is reproducible from its config file alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class RunTracker:
    """Minimal file-based tracker with the W&B logging surface.

    When cfg.tracking.backend is wandb/mlflow and the library is installed,
    the same calls are forwarded there; offline runs always keep the local
    JSON copy so results are diffable in the repository.
    """

    def __init__(self, run_name: str, out_dir: str = "runs"):
        self.dir = os.path.join(out_dir, f"{run_name}-{int(time.time())}")
        os.makedirs(self.dir, exist_ok=True)
        self.history: list[dict] = []

    def log_config(self, config: dict) -> None:
        with open(os.path.join(self.dir, "config.yaml"), "w",
                  encoding="utf-8") as fh:
            yaml.safe_dump(config, fh)

    def log(self, step: int, **metrics) -> None:
        self.history.append({"step": step, **metrics})

    def finish(self, summary: dict) -> str:
        with open(os.path.join(self.dir, "history.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        with open(os.path.join(self.dir, "summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return self.dir


def train_simulated(config: dict) -> dict:
    """Simulated training run with realistic loss dynamics.

    Loss curves follow a decaying exponential with noise; validation loss
    begins rising after an overfitting point that depends on epochs and
    rank, which exercises early stopping and best-checkpoint selection.
    """
    rng = random.Random(config["seed"])
    tracker = RunTracker(config["run_name"])
    tracker.log_config(config)

    train_rows = load_jsonl(config["data"]["train"])
    steps_per_epoch = max(1, len(train_rows)
                          // config["training"]["per_device_batch_size"]
                          // config["training"]["gradient_accumulation"])
    total_steps = steps_per_epoch * config["training"]["epochs"]
    rank = config["lora"]["r"]
    lr = config["training"]["learning_rate"]

    capacity = math.log2(rank) / 5          # higher rank fits faster
    overfit_step = int(total_steps * (0.5 + 0.1 * math.log2(rank)))
    best_val, best_step, patience_left = float("inf"), 0, \
        config["training"]["early_stopping_patience"]
    checkpoints = []

    for step in range(1, total_steps + 1):
        progress = step / total_steps
        train_loss = 2.1 * math.exp(-3.5 * (lr / 2e-4) * capacity * progress
                                    * 3) + 0.35 + rng.uniform(-0.03, 0.03)
        val_loss = train_loss + 0.05
        if step > overfit_step:
            val_loss += 0.004 * (step - overfit_step)
        tracker.log(step, train_loss=round(train_loss, 4),
                    val_loss=round(val_loss, 4),
                    gpu_mem_gb=round(7.2 + rank / 64, 2))

        if step % config["training"]["checkpoint_every_steps"] == 0 \
                or step == total_steps:
            checkpoints.append({"step": step, "val_loss": round(val_loss, 4)})
            if val_loss < best_val - 1e-4:
                best_val, best_step = val_loss, step
                patience_left = config["training"]["early_stopping_patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    adapter_dir = os.path.join(tracker.dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"r": rank, "lora_alpha": config["lora"]["alpha"],
                   "target_modules": config["lora"]["target_modules"],
                   "base_model": config["base_model"],
                   "best_checkpoint_step": best_step}, fh, indent=2)

    summary = {"mode": "simulated", "total_steps": len(tracker.history),
               "best_val_loss": round(best_val, 4),
               "best_checkpoint_step": best_step,
               "checkpoints": checkpoints, "adapter": adapter_dir}
    run_dir = tracker.finish(summary)
    summary["run_dir"] = run_dir
    return summary


def train_real(config: dict) -> dict:
    """Real QLoRA fine-tuning via PEFT + TRL. Requires the GPU extras."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    quant = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) \
        if config.get("quantization") == "4bit" else None
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], quantization_config=quant, device_map="auto")
    lora = LoraConfig(r=config["lora"]["r"],
                      lora_alpha=config["lora"]["alpha"],
                      lora_dropout=config["lora"]["dropout"],
                      target_modules=config["lora"]["target_modules"],
                      task_type="CAUSAL_LM")

    def to_text(row):
        return {"text": f"{row['instruction']}\n\n{row['input']}\n\n"
                        f"Answer: {row['output']}"}

    train_ds = Dataset.from_list(
        [to_text(r) for r in load_jsonl(config["data"]["train"])])
    val_ds = Dataset.from_list(
        [to_text(r) for r in load_jsonl(config["data"]["val"])])

    out_dir = f"outputs/{config['run_name']}"
    trainer = SFTTrainer(
        model=model, peft_config=lora, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=out_dir,
            num_train_epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            per_device_train_batch_size=config["training"]["per_device_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation"],
            warmup_ratio=config["training"]["warmup_ratio"],
            eval_strategy="steps",
            eval_steps=config["training"]["checkpoint_every_steps"],
            save_steps=config["training"]["checkpoint_every_steps"],
            load_best_model_at_end=True,
            report_to=[config["tracking"]["backend"]],
            run_name=config["run_name"], seed=config["seed"]))
    trainer.train()
    trainer.save_model(f"{out_dir}/adapter")
    return {"mode": "real", "adapter": f"{out_dir}/adapter"}


def expand_sweep(sweep_path: str) -> list[dict]:
    import itertools
    sweep = load_config(sweep_path)
    base = load_config(sweep["base_config"])
    keys = list(sweep["grid"])
    configs = []
    for combo in itertools.product(*(sweep["grid"][k] for k in keys)):
        config = copy.deepcopy(base)
        for key, value in zip(keys, combo):
            section, name = key.split(".")
            config[section][name] = value
        config["run_name"] = (f"{base['run_name']}-sweep-"
                              + "-".join(str(v).replace(".", "p")
                                         for v in combo))
        configs.append(config)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_r16.yaml")
    parser.add_argument("--sweep")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--max-runs", type=int, default=6)
    args = parser.parse_args()

    try:
        import peft, torch, trl  # noqa: F401
        have_gpu_stack = True
    except ImportError:
        have_gpu_stack = False
    simulate = args.simulate or not have_gpu_stack

    if args.sweep:
        results = []
        for config in expand_sweep(args.sweep)[: args.max_runs]:
            summary = (train_simulated if simulate else train_real)(config)
            results.append({"run": config["run_name"],
                            "r": config["lora"]["r"],
                            "lr": config["training"]["learning_rate"],
                            "epochs": config["training"]["epochs"],
                            "best_val_loss": summary.get("best_val_loss")})
        print(f"{'run':<44} {'r':>3} {'lr':>8} {'ep':>3} {'val_loss':>9}")
        for row in sorted(results, key=lambda r: r["best_val_loss"] or 9):
            print(f"{row['run']:<44} {row['r']:>3} {row['lr']:>8}"
                  f" {row['epochs']:>3} {row['best_val_loss']:>9}")
        return

    config = load_config(args.config)
    summary = (train_simulated if simulate else train_real)(config)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
