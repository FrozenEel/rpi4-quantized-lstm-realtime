"""Train the LSTM on CMAPSS FD001 and save fp32 weights + scaler + evaluation samples."""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.data import prepare, metrics
from src.model import LSTMRegressor

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="CMAPSSData")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--clip", type=int, default=125)
ap.add_argument("--out", default="artifacts")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
d = prepare(a.data, clip=a.clip)
print({k: v.shape for k, v in d.items() if hasattr(v, "shape")})

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = LSTMRegressor().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
dl = DataLoader(TensorDataset(torch.from_numpy(d["Xtr"]), torch.from_numpy(d["ytr"]).view(-1, 1)), batch_size=64, shuffle=True)

def predict(X):
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(X).to(dev)).cpu().numpy().ravel()

best = (1e9, None)
for ep in range(a.epochs):
    m.train()
    for xb, yb in dl:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss_fn(m(xb), yb).backward(); opt.step()
    v = metrics(d["yva"], predict(d["Xva"]))
    if v["rmse"] < best[0]:
        best = (v["rmse"], {k: t.detach().cpu().clone() for k, t in m.state_dict().items()})
    print(f"ep {ep+1:02d} val RMSE {v['rmse']:.2f} R2 {v['r2']:.3f}")

m.load_state_dict(best[1])
res = {"val": metrics(d["yva"], predict(d["Xva"])), "test_official": metrics(d["yte"], predict(d["Xte"]))}
print(json.dumps(res, indent=1))
torch.save(m.cpu().state_dict(), f"{a.out}/lstm_fp32.pt")
np.savez(f"{a.out}/samples.npz", Xva=d["Xva"], yva=d["yva"], Xte=d["Xte"], yte=d["yte"], mean=d["mean"], scale=d["scale"])
json.dump(res, open(f"{a.out}/train_metrics.json", "w"), indent=1)
