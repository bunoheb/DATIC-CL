#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import torch
from datetime import datetime
import shutil

from curriculum.algorithms import AdaptiveTrainer, SelfPacedTrainer
from curriculum.backbones import get_net  # Create backbones such as convnet/resnet

# -----------------------------
# Run-dir helpers
# -----------------------------
def make_run_dirs(method: str, data_name: str, variant: str, seed: int, base: str = "runs"):
    """
    runs/<method>/<data>/<variant>/seed<seed>/<timestamp>/{logs,models}
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base, method, data_name, variant, f"seed{seed}", ts)
    sub = {
        "logs":   os.path.join(run_dir, "logs"),
        "models": os.path.join(run_dir, "models"),
    }
    os.makedirs(sub["logs"], exist_ok=True)
    os.makedirs(sub["models"], exist_ok=True)
    return run_dir, sub

def save_config(path, cfg: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

# -----------------------------
# CLI
# -----------------------------
def parse_cli():
    p = argparse.ArgumentParser(description="AutoCL (ACL/SPL) — unified trainer with standardized run dirs")

    # run-dir meta
    p.add_argument("--method", type=str, default="AutoCL", help="Run method name (folder)")
    p.add_argument("--variant", type=str, choices=["ACL", "SPL"], required=True, help="AutoCL variant")
    p.add_argument("--data", type=str, default="rvl", help="Dataset name, e.g., rvl")
    p.add_argument("--net", type=str, default="convnet", help="Network, e.g., convnet, resnet34")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num-classes", type=int, default=16)

    # training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use-huggingface", action="store_true")

    # ===== ACL hyper-params =====
    p.add_argument("--pace-p", type=float, default=0.5)
    p.add_argument("--pace-q", type=float, default=2 ** (1/49))
    p.add_argument("--pace-r", type=int,   default=3)
    p.add_argument("--inv",    type=int,   default=10)
    p.add_argument("--alpha",  type=float, default=0.7)
    p.add_argument("--gamma",  type=float, default=0.1)
    p.add_argument("--gamma-decay", type=float, default=None)
    p.add_argument("--bottom-gamma", type=float, default=0.1)
    p.add_argument("--teacher", type=str, help="[ACL only] Path to pretrained teacher .pth")

    # ===== SPL hyper-params =====
    p.add_argument("--start-rate", type=float, default=0.5)
    p.add_argument("--grow-epochs", type=int, default=149)
    p.add_argument("--grow-fn", type=str, default="linear", choices=["linear", "step", "root"])
    p.add_argument("--weight-fn", type=str, default="hard", choices=["hard", "soft"])

    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_cli()

    # 1) run-dirs
    run_dir, sub = make_run_dirs(
        method=args.method, data_name=args.data, variant=args.variant, seed=args.seed
    )
    logs_dir, models_dir = sub["logs"], sub["models"]

    # 2) device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3) Generate trainer (branching based on variant)
    if args.variant == "ACL":
        if not args.teacher:
            raise ValueError("--teacher is required for ACL variant.")
        # Teacher network preparation
        teacher_net = get_net(args.net, args.data)
        teacher_state = torch.load(args.teacher, map_location="cpu")
        teacher_net.load_state_dict(teacher_state)

        trainer = AdaptiveTrainer(
            data_name=args.data,
            net_name=args.net,
            device_name=device,
            num_epochs=args.epochs,
            random_seed=args.seed,
            num_classes=args.num_classes,
            pace_p=args.pace_p,
            pace_q=args.pace_q,
            pace_r=args.pace_r,
            inv=args.inv,
            alpha=args.alpha,
            gamma=args.gamma,
            gamma_decay=args.gamma_decay,
            bottom_gamma=args.bottom_gamma,
            pretrained_net=teacher_net,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_huggingface=args.use_huggingface
        )

    else:  # SPL
        trainer = SelfPacedTrainer(
            data_name=args.data,
            net_name=args.net,
            device_name=device,
            num_epochs=args.epochs,
            random_seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            use_huggingface=args.use_huggingface,
            start_rate=args.start_rate,
            grow_epochs=args.grow_epochs,
            grow_fn=args.grow_fn,
            weight_fn=args.weight_fn
        )

    # 4) training
    trainer.fit()

    # 5) Internal output path (assuming use of Base-like internal logger)
    internal_log_dir = getattr(trainer.trainer, "log_dir", None)
    best_src  = os.path.join(internal_log_dir, "net.pkl")      if internal_log_dir else None
    last_src  = os.path.join(internal_log_dir, "last_net.pkl") if internal_log_dir else None
    trainlog  = os.path.join(internal_log_dir, "train.log")    if internal_log_dir else None

    # 6) Save to standard path
    best_dst  = os.path.join(models_dir, f"best_classifier_seed{args.seed}.pth")
    final_dst = os.path.join(models_dir, f"final_classifier_seed{args.seed}.pth")

    if best_src and os.path.exists(best_src):
        state = torch.load(best_src, map_location="cpu")
        torch.save(state, best_dst)
    else:
        torch.save(trainer.trainer.net.state_dict(), best_dst)

    if last_src and os.path.exists(last_src):
        state = torch.load(last_src, map_location="cpu")
        torch.save(state, final_dst)
    else:
        torch.save(trainer.trainer.net.state_dict(), final_dst)

    # 7) Copy log
    if trainlog and os.path.exists(trainlog):
        shutil.copy2(trainlog, os.path.join(logs_dir, "train.log"))

    # 8) Save configuration
    cfg = {
        "method": args.method,
        "variant": args.variant,
        "data": args.data,
        "net": args.net,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "use_huggingface": args.use_huggingface,
        "device": str(device),
        "internal_log_dir": internal_log_dir,
    }
    if args.variant == "ACL":
        cfg.update({
            "pace_p": args.pace_p,
            "pace_q": args.pace_q,
            "pace_r": args.pace_r,
            "inv": args.inv,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "gamma_decay": args.gamma_decay,
            "bottom_gamma": args.bottom_gamma,
            "teacher": args.teacher,
        })
    else:
        cfg.update({
            "start_rate": args.start_rate,
            "grow_epochs": args.grow_epochs,
            "grow_fn": args.grow_fn,
            "weight_fn": args.weight_fn,
        })
    save_config(os.path.join(run_dir, "config.json"), cfg)

    print("\n=== Save summary (AutoCL unified) ===")
    print(f"RUN DIR : {run_dir}")
    print(f"  ├─ logs   : {logs_dir}")
    print(f"  └─ models : {models_dir}")
    print(f"Best  -> {best_dst}")
    print(f"Final -> {final_dst}")
