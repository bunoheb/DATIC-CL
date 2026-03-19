#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import shutil
import json
from datetime import datetime
import torch

from curriculum.algorithms import BaseTrainer

# -----------------------------
# Run-dir helpers
# -----------------------------
def net_to_group(net_name: str) -> str:
    return net_name

def make_run_dirs(method: str, net_name: str, data_name: str, seed: int, base: str = "runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    net_group = net_to_group(net_name)
    run_dir = os.path.join(base, method, data_name, net_group, f"seed{seed}", ts)
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
    p = argparse.ArgumentParser(description="Baseline training (no curriculum)")
    p.add_argument("--method", type=str, default="Base", help="Method name for run dir")
    p.add_argument("--data", type=str, default="rvl", help="Dataset name (BaseTrainer supports)")
    p.add_argument("--net", type=str, default="resnet34", help="Network name (e.g., resnet34 or convnet)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--epochs", type=int, default=5, help="Num training epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use-huggingface", action="store_true", help="Use HF dataset pipeline")
    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cli = parse_cli()

    # 1) Build run dirs: runs/<method>/<data>/<net_group>/seed<seed>/<ts>/{logs,models}
    run_dir, sub = make_run_dirs(method=cli.method, net_name=cli.net, data_name=cli.data, seed=cli.seed)
    logs_dir, models_dir = sub["logs"], sub["models"]

    # 2) Train
    trainer = BaseTrainer(
        data_name=cli.data,
        net_name=cli.net,
        device_name=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        num_epochs=cli.epochs,
        random_seed=cli.seed,
        batch_size=cli.batch_size,
        learning_rate=cli.lr,
        use_huggingface=cli.use_huggingface,
    )
    trainer.fit()

    # 3) Collect and organize internal logs/checkpoints
    internal_log_dir = getattr(trainer.trainer, "log_dir", None)
    best_src  = os.path.join(internal_log_dir, "net.pkl")        if internal_log_dir else None
    last_src  = os.path.join(internal_log_dir, "last_net.pkl")   if internal_log_dir else None
    trainlog  = os.path.join(internal_log_dir, "train.log")      if internal_log_dir else None

    # best model
    best_dst = os.path.join(models_dir, f"best_classifier_seed{cli.seed}.pth")
    if best_src and os.path.exists(best_src):
        state = torch.load(best_src, map_location="cpu")
        torch.save(state, best_dst)
    else:
        torch.save(trainer.trainer.net.state_dict(), best_dst)

    # final model
    final_dst = os.path.join(models_dir, f"final_classifier_seed{cli.seed}.pth")
    if last_src and os.path.exists(last_src):
        state = torch.load(last_src, map_location="cpu")
        torch.save(state, final_dst)
    else:
        torch.save(trainer.trainer.net.state_dict(), final_dst)

    # copy train.log 
    if trainlog and os.path.exists(trainlog):
        import shutil
        shutil.copy2(trainlog, os.path.join(logs_dir, "train.log"))

    # save configuration
    save_config(os.path.join(run_dir, "config.json"), {
        "method": cli.method,
        "data": cli.data,
        "net": cli.net,
        "seed": cli.seed,
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "use_huggingface": cli.use_huggingface,
        "internal_log_dir": internal_log_dir,
    })

    print("\n=== Save summary ===")
    print(f"RUN DIR : {run_dir}")
    print(f"  ├─ logs   : {logs_dir}")
    print(f"  └─ models : {models_dir}")
    print(f"Best  -> {best_dst}")
    print(f"Final -> {final_dst}")
