"""Task 12: train a tiny flow-matching model on 2D points and animate noise turning into a shape.

Run from the 12-diffusion-explainer folder:
    ..\\.venv\\Scripts\\python flow_matching.py

The idea in four lines
  1. Data: points on a shape (two moons). Noise: points from a round Gaussian blob.
  2. Pick a noise point x0 and a data point x1. Draw the straight line between them: x_t = (1-t)*x0 + t*x1.
  3. Train a small network v(x, t) to predict the direction of travel along that line, which is x1 - x0.
  4. To generate: start from fresh noise and follow v(x, t) from t=0 to t=1 in small steps (Euler steps).

Writes results/flow_loss.csv, flow_steps.csv, flow_snapshots.png, flow_steps.png, flow_animation.gif
It uses the CPU on purpose: the model is tiny and CPU runs give the same numbers every time.
"""
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("results", exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)


def two_moons(n, noise=0.06):
    """n points on two interleaved half circles, centred and scaled to roughly [-2, 2]."""
    k = n // 2
    a = rng.uniform(0, np.pi, k)
    b = rng.uniform(0, np.pi, n - k)
    top = np.stack([np.cos(a), np.sin(a)], 1)
    bottom = np.stack([1 - np.cos(b), 0.5 - np.sin(b)], 1)
    pts = np.concatenate([top, bottom]) + rng.normal(0, noise, (n, 2))
    pts = (pts - np.array([0.5, 0.25])) * 1.6
    return pts.astype(np.float32)


class VectorField(nn.Module):
    """Small network: (x, y, t) in, direction (vx, vy) out."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t.expand(x.shape[0], 1)], 1))


model = VectorField()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
STEPS, BATCH = 4000, 512
data_pool = torch.from_numpy(two_moons(20000))

print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters for {STEPS} steps...")
losses, t0 = [], time.time()
for step in range(1, STEPS + 1):
    x1 = data_pool[torch.randint(0, len(data_pool), (BATCH,))]   # real data points
    x0 = torch.randn(BATCH, 2)                                     # noise points
    t = torch.rand(BATCH, 1)
    xt = (1 - t) * x0 + t * x1                                     # a point on the straight line
    pred = model(xt, t)
    loss = ((pred - (x1 - x0)) ** 2).mean()                        # target direction = x1 - x0
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 500 == 0:
        print(f"  step {step:>4}  loss {np.mean(losses[-500:]):.3f}")
print(f"Trained in {time.time() - t0:.0f} s")
with open("results/flow_loss.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "loss"])
    w.writerows(enumerate(losses, 1))


@torch.no_grad()
def sample(n, steps, seed=1, keep=False):
    """Follow the learned field from noise (t=0) to data (t=1) in `steps` Euler steps."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    path = [x.clone()]
    for i in range(steps):
        t = torch.tensor([[i / steps]])
        x = x + model(x, t) / steps
        if keep:
            path.append(x.clone())
    return (torch.stack(path).numpy() if keep else x.numpy())


def quality(points, ref):
    """Average distance from each generated point to its nearest real data point (smaller = better)."""
    d = np.sqrt(((points[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
    return float(d.min(1).mean())


ref = two_moons(3000)
N = 2000
print("Quality against the number of steps:")
rows = []
for s in (1, 2, 4, 8, 16, 32, 100):
    q = quality(sample(N, s), ref)
    rows.append((s, round(q, 4)))
    print(f"  {s:>3} steps: average distance to the shape {q:.4f}")
rows.append(("real data (reference)", round(quality(two_moons(N), ref), 4)))
rows.append(("pure noise (reference)", round(quality(np.random.default_rng(5).normal(size=(N, 2)).astype(np.float32), ref), 4)))
with open("results/flow_steps.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["euler_steps", "avg_distance_to_shape"])
    w.writerows(rows)

# --- static pictures -------------------------------------------------------
LIM = 3.2
BLUE, GREY = "#2a6ebb", "#9aa0a6"
path = sample(N, 100, keep=True)
fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
for ax, k in zip(axs, (0, 25, 50, 75, 100)):
    ax.scatter(path[k][:, 0], path[k][:, 1], s=3, color=BLUE, alpha=0.6)
    ax.set_title(f"t = {k / 100:.2f}")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Noise turns into two moons by following the learned field (100 steps)")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("results/flow_snapshots.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, s in zip(axs, (1, 2, 8, 100)):
    p = sample(N, s)
    ax.scatter(ref[:, 0], ref[:, 1], s=2, color=GREY, alpha=0.25)
    ax.scatter(p[:, 0], p[:, 1], s=3, color=BLUE, alpha=0.6)
    ax.set_title(f"{s} step{'s' if s > 1 else ''}")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("Same model, fewer Euler steps (grey = real data)")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("results/flow_steps.png", dpi=150)
plt.close(fig)

# --- animation: points plus the arrows of the field at that moment -----------------
FRAMES = 60
anim_path = sample(1500, FRAMES, keep=True)
gx, gy = np.meshgrid(np.linspace(-LIM, LIM, 14), np.linspace(-LIM, LIM, 14))
grid = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))
fig, ax = plt.subplots(figsize=(5.2, 5.2))


def draw(i):
    ax.clear()
    t = i / FRAMES
    with torch.no_grad():
        v = model(grid, torch.tensor([[t]])).numpy()
    ax.quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], color=GREY, alpha=0.6, scale=40, width=0.003)
    ax.scatter(anim_path[i][:, 0], anim_path[i][:, 1], s=4, color=BLUE, alpha=0.7)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Following the learned field   t = {t:.2f}")


anim = FuncAnimation(fig, draw, frames=FRAMES + 1, interval=80)
anim.save("results/flow_animation.gif", writer=PillowWriter(fps=12))
print("Saved results/flow_loss.csv, flow_steps.csv, flow_snapshots.png, flow_steps.png, flow_animation.gif")
