#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import pandas as pd

from curriculum.algorithms import BaseTrainer

# -----------------------------
# Helpers
# -----------------------------
def load_config(run_dir: str) -> Dict:
    """Read run_dir/config.json and return a dict. If not found, an empty dict.."""
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def infer_method_variant_from_path(run_dir: str) -> Tuple[str, str]:
    """
    Example of a path rule:
      runs/Base/<data>/<net_group>/seed42/<ts>/
      runs/PreCL/<data>/<func_type>/seed42/<ts>/
      runs/AutoCL/<data>/<variant>/seed42/<ts>/
    """
    parts = run_dir.replace("\\", "/").split("/")
    method, variant = "Unknown", ""
    try:
        i = parts.index("runs")
        method = parts[i+1] if len(parts) > i+1 else "Unknown"
        # PreCL: runs/PreCL/<data>/<func_type>/seed...
        # AutoCL: runs/AutoCL/<data>/<variant>/seed...
        # Base:  runs/Base/<data>/<net_group>/seed...
        if method == "PreCL":
            variant = parts[i+3] if len(parts) > i+3 else ""
        elif method == "AutoCL":
            variant = parts[i+3] if len(parts) > i+3 else ""
        elif method == "Base":
            variant = parts[i+3] if len(parts) > i+3 else ""  # net_group (convnet/resnet/etc)
    except Exception:
        pass
    return method, variant

def pick_model_file(run_dir: str, which: str) -> str:
    """
    which in {'best','final'}
    - Base/PreCL/AutoCL: run_dir/models/{best|final}_classifier_seed*.pth
    """
    which = which.lower()
    is_best = (which == "best")

    parts = run_dir.replace("\\", "/").split("/")

    models_dir = os.path.join(run_dir, "models")
    patt = "best_classifier_seed*.pth" if is_best else "final_classifier_seed*.pth"
    cands = sorted(glob.glob(os.path.join(models_dir, patt)))
    return cands[-1] if cands else ""



def discover_run_dirs(root: str,
                      method_filter: List[str],
                      data_filter: List[str],
                      variant_filter: List[str],
                      latest_only: bool) -> List[str]:
    root = root.rstrip("/\\")
    run_dirs = []

    patterns = [
        f"{root}/Base/*/*/seed*/????????-??????",
        f"{root}/PreCL/*/*/seed*/????????-??????",
        f"{root}/AutoCL/*/*/seed*/????????-??????",
    ]
    cands = []
    for p in patterns:
        cands.extend(glob.glob(p))

    # filtering
    def _ok(path: str) -> bool:
        path_u = path.replace("\\", "/")
        parts = path_u.split("/")
        ok_method = True
        ok_data = True
        ok_variant = True

        try:
            i = parts.index("runs")
            method = parts[i+1] if len(parts) > i+1 else ""
            data   = parts[i+2] if len(parts) > i+2 else ""
            # Variant location varies by method
            variant = ""
            if method == "PreCL":   # runs/PreCL/<data>/<func>/seed...
                variant = parts[i+3] if len(parts) > i+3 else ""
            elif method == "AutoCL":# runs/AutoCL/<data>/<variant>/seed...
                variant = parts[i+3] if len(parts) > i+3 else ""
            elif method == "Base":  # runs/Base/<data>/<net_group>/seed...
                variant = parts[i+3] if len(parts) > i+3 else ""

            if method_filter:
                ok_method = method in method_filter
            if data_filter:
                ok_data = data in data_filter
            if variant_filter:
                ok_variant = variant in variant_filter
            return ok_method and ok_data and ok_variant
        except Exception:
            return False

    cands = [d for d in cands if _ok(d)]

    if not latest_only:
        return sorted(cands)

    # latest_only: Only the latest by seed
    # key = (method, data, variant, seed)
    buckets: Dict[Tuple[str, str, str, str], List[str]] = {}
    for d in cands:
        d_u = d.replace("\\", "/")
        parts = d_u.split("/")
        i = parts.index("runs")
        method = parts[i+1]
        data   = parts[i+2]
        # seed folder name ex) seed42
        seed = ""
        if method in ("PreCL", "AutoCL", "Base"):
            seed = parts[i+4] if len(parts) > i+4 else ""
        variant = ""
        if method == "PreCL":
            variant = parts[i+3]
        elif method == "AutoCL":
            variant = parts[i+3]
        elif method == "Base":
            variant = parts[i+3]

        key = (method, data, variant, seed)
        buckets.setdefault(key, []).append(d)

    latest = []
    for key, arr in buckets.items():
        arr_sorted = sorted(arr)  # Sort folder names by timestamp -> Latest is last
        latest.append(arr_sorted[-1])
    return sorted(latest)

def evaluate_one_run(run_dir: str, which: str = "best") -> Dict:
    """
    Read run_dir/config.json, create a BaseTrainer, and load the models/<best|final> checkpoint to measure valid/test performance.
    """
    cfg = load_config(run_dir)
    # Mandatory key inference (may be absent, so safely default)
    data = cfg.get("data", "rvl")
    net  = cfg.get("net", "convnet")
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 64))
    lr = float(cfg.get("lr", 1e-4))
    use_hf = bool(cfg.get("use_huggingface", False))

    # BaseTrainer ready
    trainer = BaseTrainer(
        data_name=data,
        net_name=net,
        device_name=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        num_epochs=1,                 
        random_seed=seed,
        batch_size=batch_size,
        learning_rate=lr,
        use_huggingface=use_hf,
    )

    # Load model weights
    ckpt = pick_model_file(run_dir, which=which)
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"[{run_dir}] checkpoint not found for {which}")

    state = torch.load(ckpt, map_location="cpu")
    trainer.trainer.net.load_state_dict(state)

    # evaluation (test/valid on trainer)
    valid_loss, valid_acc = trainer.test(trainer.trainer.valid_loader, trainer.trainer.device)
    test_loss,  test_acc  = trainer.test(trainer.trainer.test_loader,  trainer.trainer.device)

    method, variant = infer_method_variant_from_path(run_dir)

    out = {
        "run_dir": run_dir,
        "method": cfg.get("method", method),
        "variant": cfg.get("variant", variant),
        "data": data,
        "net": net,
        "seed": seed,
        "which": which,
        "valid_loss": float(valid_loss),
        "valid_acc": float(valid_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "checkpoint": ckpt,
    }
    return out

def evaluate_one_ckpt(row: dict) -> dict:
    # 1) ckpt_path
    ckpt = row.get("ckpt_path")
    if not ckpt:
        raise ValueError("row must contain 'ckpt_path'")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    # 2) Automatic metadata inference (only if missing)
    meta = infer_meta_from_filename(ckpt)  # {method, variant, data, net, seed, which}
    method  = row.get("method",  meta["method"])
    variant = row.get("variant", meta["variant"])
    data    = row.get("data",    meta["data"])
    net     = row.get("net",     meta["net"])
    seed    = int(row.get("seed", meta["seed"]))
    which   = row.get("which",   meta.get("which", "best"))

    batch_size = int(row.get("batch_size", 64))
    lr         = float(row.get("lr", 1e-4))
    use_hf     = bool(row.get("use_huggingface", False))

    trainer = BaseTrainer(
        data_name=data,
        net_name=net,
        device_name=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        num_epochs=1,
        random_seed=seed,
        batch_size=batch_size,
        learning_rate=lr,
        use_huggingface=use_hf,
    )

    state = torch.load(ckpt, map_location="cpu")
    trainer.trainer.net.load_state_dict(state)

    valid_loss, valid_acc = trainer.test(trainer.trainer.valid_loader, trainer.trainer.device)
    test_loss,  test_acc  = trainer.test(trainer.trainer.test_loader,  trainer.trainer.device)

    return {
        "run_dir": os.path.dirname(ckpt),
        "method": method, "variant": variant,
        "data": data, "net": net, "seed": seed,
        "which": which,
        "valid_loss": float(valid_loss), "valid_acc": float(valid_acc),
        "test_loss": float(test_loss),   "test_acc": float(test_acc),
        "checkpoint": ckpt,
        "source": "external",
    }


def infer_meta_from_filename(path: str):
    name = os.path.basename(path).lower()
    import re
    m = re.search(r"_seed(\d+)", name)
    seed = int(m.group(1)) if m else 42

    # net
    if "resnet34" in name: net = "resnet34"
    elif "convnet" in name: net = "convnet"
    else: net = "convnet"

    # method/variant
    if name.startswith("acl_"):   method, variant = "AutoCL", "ACL"
    elif name.startswith("spl_"): method, variant = "AutoCL", "SPL"
    elif name.startswith("linear_"): method, variant = "PreCL", "linear"
    elif name.startswith("step_"):   method, variant = "PreCL", "step"
    elif name.startswith("root_"):   method, variant = "PreCL", "root"
    elif "resnet34" in name or "convnet" in name:
        method, variant = "Base", net
    else:
        method, variant = "Unknown", ""

    return dict(method=method, variant=variant, data="rvl", net=net, seed=seed, which="best")


def save_results(rows: List[Dict], out_root: str = "results") -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary.csv")
    json_path = os.path.join(out_dir, "summary.json")

    df.sort_values(by=["method", "variant", "data", "net", "seed", "which"], inplace=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\n===== Evaluation Summary =====")
    if not df.empty:
        cols = ["method", "variant", "data", "net", "seed", "which", "valid_acc", "test_acc"]
        print(df[cols].to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    return out_dir

# -----------------------------
# CLI
# -----------------------------
def parse_cli():
    p = argparse.ArgumentParser(description="Evaluate & compare multiple runs")
    # Usage mode 1) Directly specify run dirs
    p.add_argument("--run-dirs", nargs="*", default=[],
                   help="Evaluate these run directories directly (each should contain models/, config.json).")

    # Usage Mode 2) Auto-navigation
    p.add_argument("--root", type=str, default="runs", help="Root folder to discover runs")
    p.add_argument("--method-filter", type=str, default="", help="Comma list, e.g. Base,PreCL,AutoCL")
    p.add_argument("--data-filter", type=str, default="", help="Comma list, e.g. rvl")
    p.add_argument("--variant-filter", type=str, default="",
                   help="Comma list. For PreCL: step/linear/root, AutoCL: ACL/SPL, Base: convnet/resnet")
    p.add_argument("--latest-only", action="store_true", help="Pick only the latest run per (method,data,variant,seed)")
    p.add_argument("--manifest", type=str, default="",
                   help="CSV with columns: ckpt_path,method,variant,data,net,seed[,which]")
    p.add_argument("--ckpt-files", nargs="*", default=[],
                   help="Direct list of .pth files. Method/variant/net inferred from filename.")


    # Common Options
    p.add_argument("--which", type=str, default="best", choices=["best", "final"],
                   help="Which checkpoint to evaluate")
    p.add_argument("--out-root", type=str, default="results", help="Where to save the summary")
    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_cli()

    results = []

    # 1) Auto-detect (use the latest options)
    run_dirs = list(args.run_dirs)
    if not run_dirs:
        method_filter = [m.strip() for m in args.method_filter.split(",") if m.strip()]
        data_filter   = [d.strip() for d in args.data_filter.split(",") if d.strip()]
        variant_filter= [v.strip() for v in args.variant_filter.split(",") if v.strip()]
        run_dirs = discover_run_dirs(
            root=args.root,
            method_filter=method_filter,
            data_filter=data_filter,
            variant_filter=variant_filter,
            latest_only=args.latest_only
        )

    for rd in run_dirs:
        try:
            res = evaluate_one_run(rd, which=args.which)
            res["source"] = "discovered"
            print(f"[OK] {rd}  -> test_acc={res['test_acc']:.4f}")
            results.append(res)
        except Exception as e:
            print(f"[FAIL] {rd}: {e}")

    # 2) External checkpoint: manifest
    if args.manifest:
        import pandas as pd
        mf = pd.read_csv(args.manifest)
        for _, r in mf.iterrows():
            try:
                meta = r.to_dict()
                meta.setdefault("which", args.which)
                res = evaluate_one_ckpt(meta)
                print(f"[OK] {meta['ckpt_path']} -> test_acc={res['test_acc']:.4f}")
                results.append(res)
            except Exception as e:
                print(f"[FAIL] {r.get('ckpt_path','(unknown)')}: {e}")

    # 3) External checkpoint: file list
    if args.ckpt_files:
        for pth in args.ckpt_files:
            try:
                meta = infer_meta_from_filename(pth)
                meta["ckpt_path"] = pth
                meta.setdefault("which", args.which)
                res = evaluate_one_ckpt(meta)
                print(f"[OK] {pth} -> test_acc={res['test_acc']:.4f}")
                results.append(res)
            except Exception as e:
                print(f"[FAIL] {pth}: {e}")

    # save the results
    if results:
        save_results(results, out_root=args.out_root)
    else:
        print("Nothing evaluated.")

