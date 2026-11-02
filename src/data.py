"""CMAPSS FD001 loading, engine-wise split, sliding windows.

Differences from the original course notebook (on purpose):
  * the split is by ENGINE (not by random window), so no engine appears in both train and val
  * the scaler is fit on training engines only
  * the model is also scored on the official test set (last window of every test engine vs RUL_FD001.txt)
"""
import numpy as np
import pandas as pd

COLS = ["engine_id", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"sensor{i}" for i in range(1, 22)]
FEATS = COLS[2:]  # 24 features


def load_fd(root, fd="FD001"):
    train = pd.read_csv(f"{root}/train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(f"{root}/test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul_test = np.loadtxt(f"{root}/RUL_{fd}.txt")
    return train, test, rul_test


def add_rul(df, clip=None):
    mx = df.groupby("engine_id")["cycle"].transform("max")
    rul = (mx - df["cycle"]).astype(float)
    df = df.copy()
    df["RUL"] = rul.clip(upper=clip) if clip else rul
    return df


def windows(df, mean, scale, w):
    X, y = [], []
    for _, g in df.groupby("engine_id"):
        g = g.sort_values("cycle")
        f = ((g[FEATS].values - mean) / scale).astype(np.float32)
        r = g["RUL"].values
        for i in range(len(g) - w + 1):
            X.append(f[i:i + w]); y.append(r[i + w - 1])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def last_windows(df, mean, scale, w):
    X = []
    for _, g in df.groupby("engine_id"):
        g = g.sort_values("cycle")
        f = ((g[FEATS].values - mean) / scale).astype(np.float32)
        if len(f) < w:  # pad by repeating the first row
            f = np.vstack([np.repeat(f[:1], w - len(f), axis=0), f])
        X.append(f[-w:])
    return np.asarray(X, np.float32)


def prepare(root, w=30, clip=125, val_frac=0.2, seed=42, fd="FD001"):
    train, test, rul_test = load_fd(root, fd)
    train = add_rul(train, clip)
    ids = train["engine_id"].unique()
    rng = np.random.RandomState(seed)
    val_ids = set(rng.choice(ids, size=int(round(len(ids) * val_frac)), replace=False))
    tr = train[~train["engine_id"].isin(val_ids)]
    va = train[train["engine_id"].isin(val_ids)]
    mean = tr[FEATS].values.mean(0)
    scale = tr[FEATS].values.std(0)
    scale[scale < 1e-8] = 1.0  # constant sensors
    Xtr, ytr = windows(tr, mean, scale, w)
    Xva, yva = windows(va, mean, scale, w)
    Xte = last_windows(test, mean, scale, w)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=rul_test.astype(np.float32),
                mean=mean, scale=scale)


def metrics(y, p):
    d = p - y
    rmse = float(np.sqrt(np.mean(d ** 2)))
    mae = float(np.mean(np.abs(d)))
    r2 = float(1 - np.sum(d ** 2) / np.sum((y - y.mean()) ** 2))
    # NASA asymmetric score: late predictions (d>0) are penalised more
    score = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
    return dict(rmse=rmse, mae=mae, r2=r2, nasa_score=score)
