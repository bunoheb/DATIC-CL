#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import torch
from datetime import datetime

from curriculum.algorithms import PredefinedTrainer

# -----------------------------
# Run-dir helpers
# -----------------------------
def make_run_dirs(method: str, data_name: str, func_type: str, seed: int, base: str = "runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base, method, data_name, func_type, f"seed{seed}", ts)
    sub = {
        "logs":   os.path.join(run_dir, "logs"),
        "models": os.path.join(run_dir, "models"),
    }
    os.makedirs(sub["logs"], exist_ok=True)
    os.makedirs(sub["models"], exist_ok=True)
    return run_dir, sub

def save_config(path, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# -----------------------------
# CLI
# -----------------------------
def parse_cli():
    p = argparse.ArgumentParser(description="Predefined Curriculum Learning (PreCL) trainer")
    # run-dir meta
    p.add_argument("--method", type=str, default="PreCL", help="Method name for run dir")
    p.add_argument("--data", type=str, default="rvl", help="Dataset name")
    p.add_argument("--net", type=str, default="convnet", help="Network (e.g., convnet, resnet34)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use-huggingface", action="store_true")

    # curriculum schedule
    p.add_argument("--func-type", type=str, default="step", choices=["step", "linear", "root"],
                   help="Schedule type (Reflected at folder level)")
    p.add_argument("--num-steps", type=int, default=50, help="Number of curriculum steps")
    p.add_argument("--epochs-per-step", type=int, default=3,
                   help="Epochs per step (Mainly used in step schedules)")

    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cli = parse_cli()

    # 1) Run dirs: runs/PreCL/<data>/<func_type>/seed<seed>/<ts>/{logs,models}
    run_dir, sub = make_run_dirs(
        method=cli.method,
        data_name=cli.data,
        func_type=cli.func_type,
        seed=cli.seed,
    )
    logs_dir, models_dir = sub["logs"], sub["models"]

    # 2) Trainer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = PredefinedTrainer(
        data_name=cli.data,
        net_name=cli.net,
        device_name=device,
        num_epochs=cli.epochs,
        random_seed=cli.seed,
        batch_size=cli.batch_size,
        learning_rate=cli.lr,
        use_huggingface=cli.use_huggingface,
        schedule_type=cli.func_type,
        num_steps=cli.num_steps,
        epochs_per_step=cli.epochs_per_step
    )

    # 3) Fit
    trainer.fit()

    # 4) Internal output path (PredefinedTrainer internal log directory)
    internal_log_dir = getattr(trainer.trainer, "log_dir", None)
    best_src  = os.path.join(internal_log_dir, "net.pkl")        if internal_log_dir else None
    last_src  = os.path.join(internal_log_dir, "last_net.pkl")   if internal_log_dir else None
    trainlog  = os.path.join(internal_log_dir, "train.log")      if internal_log_dir else None

    # 5) best / final save -> standard path
    best_dst  = os.path.join(models_dir, f"best_classifier_seed{cli.seed}.pth")
    final_dst = os.path.join(models_dir, f"final_classifier_seed{cli.seed}.pth")

    if best_src and os.path.exists(best_src):
        state = torch.load(best_src, map_location="cpu")
        torch.save(state, best_dst)
    else:
        # fallback: Save current network weights
        torch.save(trainer.trainer.net.state_dict(), best_dst)

    if last_src and os.path.exists(last_src):
        state = torch.load(last_src, map_location="cpu")
        torch.save(state, final_dst)
    else:
        torch.save(trainer.trainer.net.state_dict(), final_dst)

    # 6) copy train.log 
    if trainlog and os.path.exists(trainlog):
        import shutil
        shutil.copy2(trainlog, os.path.join(logs_dir, "train.log"))

    # 7) save configuration
    save_config(os.path.join(run_dir, "config.json"), {
        "method": cli.method,
        "data": cli.data,
        "net": cli.net,
        "seed": cli.seed,
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "use_huggingface": cli.use_huggingface,
        "func_type": cli.func_type,
        "num_steps": cli.num_steps,
        "epochs_per_step": cli.epochs_per_step,
        "internal_log_dir": internal_log_dir,
        "device": str(device),
    })

    print("\n=== Save summary (PreCL) ===")
    print(f"RUN DIR : {run_dir}")
    print(f"  ├─ logs   : {logs_dir}")
    print(f"  └─ models : {models_dir}")
    print(f"Best  -> {best_dst}")
    print(f"Final -> {final_dst}")
