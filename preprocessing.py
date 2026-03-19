#!/usr/bin/env python
# coding: utf-8
"""
End-to-end preprocessing pipeline

Steps
1) Hand-crafted features (Edge/HOG/GLCM)  -> data/new_textual_image_features.csv
2) ResNet50 embeddings (GAP-2048)         -> data/new_img_embeddings.csv
3) Merge + cosine similarity               -> data/new_img_embeddings_with_cosine.csv
4) Normalize + weighted aggregation        -> data/new_combined_features_with_cosine.csv
5) Pretrained loss with ResNet18           -> data/new_data_with_score_and_loss.csv
6) Rank-fusion final difficulty            -> data/data_with_combined_difficulty.csv
"""

import os
import gc
import json
import warnings
warnings.filterwarnings("ignore")

# ---------------- Imports ----------------
# Force TF on CPU (avoid TF↔Torch GPU contention). Comment if you want TF on GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# skimage features
from skimage.feature import hog
from skimage.feature.texture import graycomatrix, graycoprops

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# SciPy
from scipy.stats import dirichlet

# TF (ResNet50 for embeddings)
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Torch (pretrained loss)
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ------------- Utils -------------
def ensure_dir(p):
    if p and len(p) > 0:
        os.makedirs(p, exist_ok=True)

def to_abs_path(base_dir, p):
    """OS-independent path resolution (handles absolute/relative mixed)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    p = str(p)
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p.lstrip("/\\"))
    return os.path.normpath(p)


# ------------- (1) Hand-crafted features -------------
def edge_pixel_count(image_gray_uint8):
    edges = cv2.Canny(image_gray_uint8, 100, 200)
    return int(np.sum(edges > 0))

def hog_feature_mean(image_gray_uint8):
    # visualize=False for speed; returns 1D feature vector
    feats = hog(
        image_gray_uint8,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False,
        feature_vector=True,
    )
    return float(np.mean(feats)) if feats is not None and len(feats) > 0 else np.nan

def glcm_contrast_homogeneity(image_gray_uint8):
    # image should be uint8 [0..255]
    glcm = graycomatrix(image_gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    return contrast, homogeneity

def process_images_for_features(df, base_dir, output_csv="data/new_textual_image_features.csv"):
    """
    df must contain column 'path'
    """
    edge_counts, hog_means, glcm_contrasts, glcm_homos = [], [], [], []

    for i in tqdm(range(len(df)), desc="(1) Hand-crafted features"):
        img_path = to_abs_path(base_dir, df.iloc[i]['path'])
        if img_path is None or not os.path.isfile(img_path):
            print(f"[warn] Missing file: {img_path}")
            edge_counts.append(np.nan); hog_means.append(np.nan); glcm_contrasts.append(np.nan); glcm_homos.append(np.nan)
            continue
        try:
            img = Image.open(img_path).convert('L')   # grayscale
            img_np = np.array(img).astype(np.uint8)

            edge_counts.append(edge_pixel_count(img_np))
            hog_means.append(hog_feature_mean(img_np))
            c, h = glcm_contrast_homogeneity(img_np)
            glcm_contrasts.append(c)
            glcm_homos.append(h)
        except Exception as e:
            print(f"[error] {img_path}: {e}")
            edge_counts.append(np.nan); hog_means.append(np.nan); glcm_contrasts.append(np.nan); glcm_homos.append(np.nan)

    out = df.copy()
    out['edge_count']      = edge_counts
    out['hog_mean']        = hog_means
    out['glcm_contrast']   = glcm_contrasts
    out['glcm_homogeneity']= glcm_homos

    ensure_dir(os.path.dirname(output_csv))
    out.to_csv(output_csv, index=False)
    return out


# ------------- (2) ResNet50 Embeddings (TF-Keras) -------------
def build_resnet50_gap():
    base = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))  # output 2048-D
    return model

def get_embedding(model, img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB").resize(target_size)
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # IMPORTANT: proper preprocessing for ResNet50
    feats = model.predict(x, verbose=0)
    return feats.flatten()  # (2048,)

def extract_embeddings_from_dataframe(model, df, base_dir, target_size=(224, 224)):
    embeddings, processed_paths, missing = [], [], []
    for i in tqdm(range(len(df)), desc="(2) ResNet50 embeddings", unit="file"):
        img_path = to_abs_path(base_dir, df.iloc[i]['path'])
        if img_path is None or not os.path.isfile(img_path):
            missing.append(img_path); continue
        try:
            emb = get_embedding(model, img_path, target_size=target_size)
            embeddings.append(emb)
            processed_paths.append(df.iloc[i]['path'])  # keep original path for joining later
        except Exception as e:
            print(f"[error] Embedding {img_path}: {e}")
            missing.append(img_path)
    if missing:
        print(f"[info] Missing files: {len(missing)}")
    return embeddings, processed_paths

def save_embeddings_to_csv(embeddings, original_paths, output_csv="data/new_img_embeddings.csv"):
    if len(embeddings) == 0:
        print("[warn] No embeddings to save.")
        return
    emb_list = [e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings]
    dim = len(emb_list[0])
    emb_cols = {f"embedding_{i}": [e[i] for e in emb_list] for i in range(dim)}

    df = pd.DataFrame({"path": original_paths})
    for k, v in emb_cols.items():
        df[k] = v

    ensure_dir(os.path.dirname(output_csv))
    df.to_csv(output_csv, index=False)
    print(f"[ok] Embeddings saved -> {output_csv}")


# ------------- (3) Cosine similarity vs class average -------------
def merge_label_with_embeddings(embedding_csv, label_csv, embedding_prefix="embedding_"):
    emb_df = pd.read_csv(embedding_csv)
    lab_df = pd.read_csv(label_csv).rename(columns={'path': 'label_path'})
    merged = emb_df.merge(lab_df[['label_path', 'label']], left_on='path', right_on='label_path', how='inner').drop(columns=['label_path'])
    emb_cols = [c for c in merged.columns if c.startswith(embedding_prefix)]
    if not emb_cols:
        raise ValueError(f"No embedding_* columns in {embedding_csv}")
    return merged, emb_cols

def validate_and_convert_embeddings(embedding_df, embedding_cols, expected_dim=2048):
    X = embedding_df[embedding_cols].values
    valid = []
    for i, e in enumerate(X):
        if len(e) == expected_dim and not np.isnan(e).any():
            valid.append(e)
        else:
            print(f"[warn] Invalid embedding at idx={i} -> zero vector")
            valid.append(np.zeros(expected_dim, dtype=np.float32))
    return np.array(valid, dtype=np.float32)

def calculate_class_average_embeddings(embeddings, labels, expected_dim=2048):
    labels = list(labels)
    uni = sorted(set(labels))
    class_emb = {}
    for lb in uni:
        idx = [i for i, l in enumerate(labels) if l == lb]
        vecs = np.array([embeddings[i] for i in idx if embeddings[i].shape[0]==expected_dim])
        if vecs.size == 0:
            class_emb[lb] = np.zeros(expected_dim, dtype=np.float32)
        else:
            class_emb[lb] = vecs.mean(axis=0)
    return class_emb

def calculate_cosine_similarity(embeddings, labels, class_avg_embeddings, embedding_dim=2048):
    scores = []
    empty_count, max_warnings = 0, 10
    for i in range(len(embeddings)):
        lb = labels[i]
        x = embeddings[i].astype(np.float32).reshape(1, -1)
        if x.size == 0 or np.all(x == 0):
            if empty_count < max_warnings:
                print(f"[warn] Empty embedding at i={i}, label={lb} -> similarity=0")
            empty_count += 1
            scores.append(0.0); continue
        c = class_avg_embeddings.get(lb, np.zeros(embedding_dim, dtype=np.float32)).reshape(1, -1)
        sim = float(cosine_similarity(x, c)[0,0])
        scores.append(sim)
    if empty_count > max_warnings:
        print(f"[info] Additional empty embeddings: {empty_count - max_warnings} (total {empty_count})")
    return scores

def process_and_save_cosine_similarity(embedding_csv, label_csv, output_csv, embedding_prefix="embedding_", expected_dim=2048):
    print("[step] Loading & merging for cosine similarity...")
    merged, emb_cols = merge_label_with_embeddings(embedding_csv, label_csv, embedding_prefix)
    print(f"[info] merged rows: {len(merged)}, embedding dims: {len(emb_cols)}")
    embeddings = validate_and_convert_embeddings(merged, emb_cols, expected_dim)
    labels = merged['label'].tolist()
    print("[step] Compute class means...")
    class_means = calculate_class_average_embeddings(embeddings, labels, expected_dim)
    print("[step] Compute cosine similarity...")
    sims = calculate_cosine_similarity(embeddings, labels, class_means, expected_dim)
    merged['img_cos_sim'] = sims
    ensure_dir(os.path.dirname(output_csv))
    merged.to_csv(output_csv, index=False)
    print(f"[ok] Cosine similarity saved -> {output_csv}")
    return merged


# ------------- (4) Normalization + weighted aggregation -------------
def optimize_weights_dirichlet(features, target, alpha, n_samples=10000, random_state=42):
    """
    Simple random search with Dirichlet sampling to find weights minimizing MSE to target.
    features: (N, K), target: (N,), alpha: (K,)
    """
    rng = np.random.default_rng(random_state)
    best_w, best_score = None, float('inf')
    alpha = np.asarray(alpha, dtype=float)
    for _ in range(n_samples):
        w = rng.dirichlet(alpha)  # (K,)
        y = features @ w
        sc = mean_squared_error(target, y)
        if sc < best_score:
            best_score, best_w = sc, w
    return best_w, best_score

def normalize_and_aggregate(textual_features_csv, img_cosine_csv, output_combined_csv):
    # Load
    tdf = pd.read_csv(textual_features_csv)
    idf = pd.read_csv(img_cosine_csv)[['path', 'img_cos_sim']] if 'img_cos_sim' in pd.read_csv(img_cosine_csv).columns \
        else pd.read_csv(img_cosine_csv)[['path', 'cosine_similarity']].rename(columns={'cosine_similarity':'img_cos_sim'})
    # Merge
    tdf = tdf[['path','label','edge_count','hog_mean','glcm_contrast','glcm_homogeneity']]
    comb = tdf.merge(idf, on='path', how='left')

    # a) img_cos_sim [-1,1] → [0,1] if needed
    if comb['img_cos_sim'].min() < 0.0:
        comb['img_cos_sim'] = (comb['img_cos_sim'] + 1.0) / 2.0

    # b) NaN handling before scaling
    num_cols = ['edge_count','hog_mean','glcm_contrast','glcm_homogeneity','img_cos_sim']
    comb[num_cols] = comb[num_cols].astype(float)
    comb = comb.dropna(subset=num_cols)  # simplest safe option

    # c) Standard scaling for 4 textual features
    scaler = StandardScaler()
    comb[['edge_count','hog_mean','glcm_contrast','glcm_homogeneity']] = scaler.fit_transform(
        comb[['edge_count','hog_mean','glcm_contrast','glcm_homogeneity']]
    )

    # d) Dirichlet weight search
    X = comb[['edge_count','hog_mean','glcm_contrast','glcm_homogeneity','img_cos_sim']].values  # (N,5)
    # Here we don't have a true target, so we use a dummy target (or any heuristic)
    # If you have a real target 'difficulty', plug it here.
    target = np.random.rand(X.shape[0])  # placeholder
    alpha = [1.0] * X.shape[1]
    w, score = optimize_weights_dirichlet(X, target, alpha, n_samples=5000, random_state=42)
    print(f"[info] best weights: {np.round(w,4)}  (mse={score:.6f})")

    # e) difficulty score and normalize to [0,1]
    d_raw = X @ w
    mm = MinMaxScaler()
    d_norm = mm.fit_transform(d_raw.reshape(-1,1)).flatten()

    out = comb.copy()
    out['score'] = d_norm

    ensure_dir(os.path.dirname(output_combined_csv))
    out.to_csv(output_combined_csv, index=False)
    print(f"[ok] Combined features saved -> {output_combined_csv}")
    return out


# ------------- (5) Pretrained loss (ResNet18) -------------
class PreTestDataset(Dataset):
    def __init__(self, csv_path):
        self.data_df = pd.read_csv(csv_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx]['path']
        label = int(self.data_df.iloc[idx]['label'])  # ensure int64 below
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label, idx
        except Exception as e:
            print(f"[warn] Load failed at idx={idx}, {img_path}: {e}")
            return torch.zeros((3,224,224)), -1, idx

def calculate_pretrain_loss(input_csv, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PreTestDataset(input_csv)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    pin_memory=torch.cuda.is_available())

    # torchvision API (weights instead of pretrained=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 16)
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    loss_dict = {}
    with torch.no_grad():
        for images, labels, indices in tqdm(dl, desc="(5) Pretrained loss"):
            images = images.to(device)
            labels = labels.to(device).long()  # ensure Long for CE

            outputs = model(images)
            # mask invalid labels
            mask = labels != -1
            if mask.any():
                losses = criterion(outputs[mask], labels[mask])
                idxs = indices[mask.cpu()].numpy()
                for k, lv in zip(idxs, losses.detach().cpu().numpy()):
                    loss_dict[int(k)] = float(lv)

            del images, labels, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()

    df = pd.read_csv(input_csv)
    df['pre_loss'] = df.index.map(lambda x: loss_dict.get(x, np.nan))

    ensure_dir(os.path.dirname(output_csv))
    df.to_csv(output_csv, index=False)
    print(f"[ok] Loss saved -> {output_csv}")


# ------------- (6) Rank fusion -------------
def combine_metrics_by_rank(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # rank on available values only
    score_rank = df['score'].rank(pct=True)
    loss_rank  = df['pre_loss'].rank(pct=True)
    combined   = (score_rank + loss_rank) / 2.0

    # quick visualization is omitted in .py (no interactive)
    df['difficulty'] = combined
    ensure_dir(os.path.dirname(output_csv))
    df.to_csv(output_csv, index=False)
    print(f"[ok] Difficulty saved -> {output_csv}")
    return df


# ------------- Orchestration -------------
def main():
    # I/O paths (adjust if needed)
    base_dir = ""  # root for relative paths in CSV
    data_dir = "data"
    ensure_dir(data_dir)

    # Input CSV containing columns: path,label
    csv_paths = os.path.join(data_dir, "data_path.csv")

    # (1) features
    textual_out = os.path.join(data_dir, "new_textual_image_features.csv")
    df = pd.read_csv(csv_paths)
    df_feat = process_images_for_features(df, base_dir, output_csv=textual_out)

    # (2) embeddings
    emb_out = os.path.join(data_dir, "new_img_embeddings.csv")
    model = build_resnet50_gap()
    embs, ok_paths = extract_embeddings_from_dataframe(model, df, base_dir)
    if len(embs) > 0:
        save_embeddings_to_csv(embs, ok_paths, output_csv=emb_out)
    else:
        print("[warn] No embeddings extracted. Skipping next steps may fail.")

    # (3) cosine sim
    emb_cos_out = os.path.join(data_dir, "new_img_embeddings_with_cosine.csv")
    merged = process_and_save_cosine_similarity(emb_out, textual_out, emb_cos_out,
                                                embedding_prefix="embedding_", expected_dim=2048)

    # (4) normalize + aggregate
    combined_out = os.path.join(data_dir, "new_combined_features_with_cosine.csv")
    comb_df = normalize_and_aggregate(textual_out, emb_cos_out, combined_out)

    # (5) pretrained loss
    loss_out = os.path.join(data_dir, "new_data_with_score_and_loss.csv")
    calculate_pretrain_loss(combined_out, loss_out)

    # (6) final difficulty (rank fusion)
    final_out = os.path.join(data_dir, "data_with_combined_difficulty.csv")
    combine_metrics_by_rank(loss_out, final_out)

    print("\n[done] Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
