#!/usr/bin/env python3
import gc
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import trange

# ================================================================
# COMMAND-LINE ARGUMENTS
# ================================================================
if len(sys.argv) < 4:
    raise ValueError(
        "Usage: python egnn_fast_multisize_heldout.py "
        "DATA_DIR OUTPUT_DIR N_DATA [SEED]"
    )

data_dir = sys.argv[1]
output_dir = sys.argv[2]
N_data_requested = int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

if N_data_requested <= 0:
    raise ValueError(f"N_DATA must be positive, got {N_data_requested}")

os.makedirs(output_dir, exist_ok=True)

print(f"Loading trajectory data from: {data_dir}")
print(f"Saving ML outputs to:        {output_dir}")
print(f"Requested train/test size:   {N_data_requested} each")
print(f"Split seed:                  {seed}")

# ================================================================
# LOAD DATA AS MEMMAP AND CREATE DISJOINT TRAIN/TEST SETS
# ================================================================
traj_fwd_all = np.load(
    os.path.join(data_dir, "traj_fwd.npy"), mmap_mode="r"
)
traj_rev_all = np.load(
    os.path.join(data_dir, "traj_rev.npy"), mmap_mode="r"
)

print("Full traj_fwd shape:", traj_fwd_all.shape)
print("Full traj_rev shape:", traj_rev_all.shape)

if traj_fwd_all.shape != traj_rev_all.shape:
    raise ValueError(
        f"Forward/reverse trajectory shapes do not match: "
        f"{traj_fwd_all.shape} vs {traj_rev_all.shape}"
    )

if traj_fwd_all.ndim != 4:
    raise ValueError(
        "Expected trajectory shape (N, n_store, n_beads, 3). "
        f"Got {traj_fwd_all.shape}"
    )

N_all = traj_fwd_all.shape[0]
required = 2 * N_data_requested
if required > N_all:
    raise ValueError(
        f"Requested N_train=N_test={N_data_requested}, which requires "
        f"{required} trajectories, but only N_all={N_all} are available."
    )

# Reproducible disjoint split. The same indices are used for the
# corresponding forward and reverse trajectory arrays.
rng = np.random.default_rng(seed)
selected_indices = rng.choice(N_all, size=required, replace=False)
train_indices = selected_indices[:N_data_requested]
test_indices = selected_indices[N_data_requested:]

# Materialize only the selected subsets. Advanced indexing creates arrays
# containing the requested trajectories rather than loading the full dataset.
traj_fwd_train = np.asarray(traj_fwd_all[train_indices])
traj_rev_train = np.asarray(traj_rev_all[train_indices])
traj_fwd_test = np.asarray(traj_fwd_all[test_indices])
traj_rev_test = np.asarray(traj_rev_all[test_indices])

N_train = traj_fwd_train.shape[0]
N_eval = traj_fwd_test.shape[0]
_, n_store, n_beads, coord_dim = traj_fwd_train.shape
L = n_store - 1

dt = 1e-5
store_stride = 500
dt_inf = store_stride * dt
t_max = (L - 1) * dt_inf

print("Training traj_fwd shape:", traj_fwd_train.shape)
print("Test traj_fwd shape:    ", traj_fwd_test.shape)
print("Training traj_rev shape:", traj_rev_train.shape)
print("Test traj_rev shape:    ", traj_rev_test.shape)
print(
    f"dt={dt}, dt_inf={dt_inf}, L={L}, "
    f"N_train={N_train}, N_test={N_eval}, "
    f"n_beads={n_beads}, coord_dim={coord_dim}, t_max={t_max:.6f}"
)

# Save the split immediately so every result can be reproduced.
np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
np.save(os.path.join(output_dir, "test_indices.npy"), test_indices)
np.save(os.path.join(output_dir, "split_seed.npy"), np.array([seed], dtype=np.int64))

# ================================================================
# DEVICE
# ================================================================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

CACHE_DATA_ON_GPU = True

# ================================================================
# ML PARAMETERS
# ================================================================
epochs = 10000
lr = 1e-2
eps_var = 1e-6

K = 30
dim_h = 32
num_blocks = 2

traj_batch = min(5000, N_train)
time_batch = min(10, L)

eval_traj_chunk = min(1000, N_eval)
warmup_epochs = 0

# ================================================================
# MODEL
# ================================================================
def build_edges(n_beads):
    edges = []
    native_pairs = [(3, 6), (2, 7), (1, 8)]

    for i in range(n_beads - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    for i in range(n_beads - 2):
        edges.append((i, i + 2))
        edges.append((i + 2, i))

    for i, j in native_pairs:
        if i < n_beads and j < n_beads:
            edges.append((i, j))
            edges.append((j, i))

    edges = sorted(set(edges))

    senders = torch.tensor([e[0] for e in edges], dtype=torch.long)
    receivers = torch.tensor([e[1] for e in edges], dtype=torch.long)
    return senders, receivers


class TimeEmbedding(nn.Module):
    def __init__(self, K, t_max):
        super().__init__()
        mu_init = torch.linspace(0.0, t_max, K)
        self.mu = nn.Parameter(mu_init)

        init_sig = max(t_max / max(K, 1), 1e-4)
        self.log_sig = nn.Parameter(
            torch.full((K,), np.log(init_sig), dtype=torch.float32)
        )

    def forward(self, t):
        diff = t.unsqueeze(-1) - self.mu
        sig = self.log_sig.exp().clamp(min=1e-4)
        return torch.exp(-0.5 * (diff / sig) ** 2)


class EGNNLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * h + 1, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

    def forward(self, node_h, x, senders, receivers):
        xi = x[:, :, senders, :]
        xj = x[:, :, receivers, :]
        hi = node_h[:, :, senders, :]
        hj = node_h[:, :, receivers, :]

        rij = xi - xj
        dist2 = (rij ** 2).sum(dim=-1, keepdim=True)

        edge_input = torch.cat([hi, hj, dist2], dim=-1)
        mij = self.edge_mlp(edge_input)

        coord_weight = self.coord_mlp(mij)
        dx = rij * coord_weight

        coord_update = torch.zeros_like(x)
        coord_update.index_add_(2, receivers, dx)
        x = x + coord_update / max(len(senders), 1)

        msg = torch.zeros_like(node_h)
        msg.index_add_(2, receivers, mij)
        node_h = node_h + self.node_mlp(torch.cat([node_h, msg], dim=-1))

        return node_h, x


class EGNNGaussianNet(nn.Module):
    def __init__(self, n_beads, coord_dim, K, h, blocks, t_max):
        super().__init__()
        self.n_beads = n_beads
        self.coord_dim = coord_dim
        self.K = K

        self.time_emb = TimeEmbedding(K, t_max)
        self.bead_emb = nn.Embedding(n_beads, h)
        self.time_lift = nn.Linear(K, h)
        self.layers = nn.ModuleList([EGNNLayer(h) for _ in range(blocks)])
        self.out = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, coord_dim),
        )

        senders, receivers = build_edges(n_beads)
        self.register_buffer("senders", senders)
        self.register_buffer("receivers", receivers)

    @property
    def mu(self):
        return self.time_emb.mu

    @property
    def log_sig(self):
        return self.time_emb.log_sig

    def phi(self, t):
        return self.time_emb(t)

    def forward(self, x, t):
        B, M, V, C = x.shape
        bead_ids = torch.arange(V, device=x.device)

        bead_h = self.bead_emb(bead_ids)
        bead_h = bead_h.view(1, 1, V, -1).expand(B, M, V, -1)

        time_h = self.time_lift(self.phi(t))
        time_h = time_h.unsqueeze(2).expand(B, M, V, -1)

        node_h = bead_h + time_h
        x_work = x

        for layer in self.layers:
            node_h, x_work = layer(node_h, x_work, self.senders, self.receivers)

        return self.out(node_h)


def make_network():
    net = EGNNGaussianNet(
        n_beads=n_beads,
        coord_dim=coord_dim,
        K=K,
        h=dim_h,
        blocks=num_blocks,
        t_max=t_max,
    ).to(device)

    opt = optim.Adam(net.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=1e-5,
    )
    return net, opt, sched


net_fwd, opt_fwd, sched_fwd = make_network()
net_rev, opt_rev, sched_rev = make_network()

# ================================================================
# HELPERS
# ================================================================
def traj_to_mid_diff(traj, label):
    mid = 0.5 * (traj[:, :-1, :, :] + traj[:, 1:, :, :])
    diff = traj[:, 1:, :, :] - traj[:, :-1, :, :]

    mid_t = torch.from_numpy(mid.astype(np.float32))
    diff_t = torch.from_numpy(diff.astype(np.float32))

    print(
        f"  {label}: mid/diff shape = {tuple(mid_t.shape)} "
        f"({mid_t.numel() * mid_t.element_size() / 1e6:.0f} MB each)"
    )

    if CACHE_DATA_ON_GPU and device.type == "cuda":
        try:
            mid_t = mid_t.to(device, non_blocking=True)
            diff_t = diff_t.to(device, non_blocking=True)
            print(f"  {label}: cached mid/diff on GPU")
        except RuntimeError as exc:
            print(f"  {label}: GPU cache failed, keeping data on CPU")
            print(f"  Reason: {exc}")
            mid_t = mid_t.cpu()
            diff_t = diff_t.cpu()
            torch.cuda.empty_cache()

    return mid_t, diff_t


def make_time_axis(L, reverse=False):
    t = torch.arange(L, device=device, dtype=torch.float32) * dt_inf
    return t.flip(0) if reverse else t


time_fwd = make_time_axis(L, reverse=False)
time_rev = make_time_axis(L, reverse=True)


def stratified_time_sample(n):
    return random.sample(range(L), min(n, L))


def compute_jj(net, midyn_t, diffyn_t, time_axis, traj_idx, time_idx):
    if not torch.is_tensor(traj_idx):
        traj_idx = torch.tensor(traj_idx, dtype=torch.long)

    if midyn_t.device.type == "cuda":
        traj_idx = traj_idx.to(midyn_t.device, non_blocking=True)
        dm = midyn_t[traj_idx][:, time_idx, :, :]
        dx = diffyn_t[traj_idx][:, time_idx, :, :]
    else:
        dm = midyn_t[traj_idx][:, time_idx, :, :].to(device, non_blocking=True)
        dx = diffyn_t[traj_idx][:, time_idx, :, :].to(device, non_blocking=True)

    t = time_axis[time_idx].unsqueeze(0).expand(dm.shape[0], -1)
    out = net(dm, t)
    return (out * dx).sum(dim=(2, 3))

# ================================================================
# TRAINING
# ================================================================
def train(net, opt, sched, midyn_t, diffyn_t, time_axis, label):
    net.train()
    ep_loss = []

    for epoch in trange(epochs, desc=f"Training {label}"):
        if midyn_t.device.type == "cuda":
            traj_idx = torch.randperm(N_train, device=midyn_t.device)[:traj_batch]
        else:
            traj_idx = torch.randperm(N_train)[:traj_batch]

        time_idx = stratified_time_sample(time_batch)
        jj = compute_jj(net, midyn_t, diffyn_t, time_axis, traj_idx, time_idx)

        mean_jj = jj.mean(dim=0)
        var_jj = jj.var(dim=0, unbiased=True)

        if epoch < warmup_epochs:
            loss = -mean_jj.sum()
        else:
            loss = (-2.0 * mean_jj**2 / (dt_inf * (var_jj + eps_var))).sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

        ep_loss.append(float(loss.detach().cpu()))

    return ep_loss

# ================================================================
# EVALUATION OVER THE INDEPENDENT HELD-OUT TEST SET
# ================================================================
def eval_sigma_from_raw(net, traj_data, time_axis, label):
    """Evaluate sigma(t) over the held-out trajectories only."""
    net.eval()

    n_eval_local = traj_data.shape[0]
    sigma = np.zeros(L, dtype=np.float32)

    with torch.inference_mode():
        for t_idx in trange(
            L,
            desc=f"Eval {label} over all {n_eval_local} trajectories",
        ):
            welf_n = 0
            welf_mean = torch.zeros(1, device=device, dtype=torch.float32)
            welf_M2 = torch.zeros(1, device=device, dtype=torch.float32)

            for tr_start in range(0, n_eval_local, eval_traj_chunk):
                tr_end = min(tr_start + eval_traj_chunk, n_eval_local)

                x0_np = traj_data[tr_start:tr_end, t_idx, :, :].astype(np.float32)
                x1_np = traj_data[tr_start:tr_end, t_idx + 1, :, :].astype(np.float32)

                mid_np = 0.5 * (x0_np + x1_np)
                diff_np = x1_np - x0_np

                mid_t = torch.from_numpy(mid_np).to(device, non_blocking=True)
                diff_t = torch.from_numpy(diff_np).to(device, non_blocking=True)

                mid_t = mid_t[:, None, :, :]
                diff_t = diff_t[:, None, :, :]

                t = time_axis[t_idx].view(1, 1).expand(mid_t.shape[0], 1)
                out = net(mid_t, t)
                jj = (out * diff_t).sum(dim=(2, 3))

                chunk_n = jj.shape[0]
                chunk_mean = jj.mean(dim=0)

                if chunk_n > 1:
                    chunk_M2 = jj.var(dim=0, unbiased=True) * (chunk_n - 1)
                else:
                    chunk_M2 = torch.zeros_like(chunk_mean)

                new_n = welf_n + chunk_n
                delta = chunk_mean - welf_mean

                welf_mean = welf_mean + delta * chunk_n / new_n
                welf_M2 = (
                    welf_M2
                    + chunk_M2
                    + delta**2 * welf_n * chunk_n / new_n
                )
                welf_n = new_n

                del x0_np, x1_np, mid_np, diff_np
                del mid_t, diff_t, t, out, jj

            var_jj = welf_M2 / max(welf_n - 1, 1)
            sigma[t_idx] = (
                2.0 * welf_mean**2 / (dt_inf * (var_jj + eps_var))
            ).detach().cpu().item()

    return sigma

# ================================================================
# FORWARD
# ================================================================
print("\n=== Reconstructing forward mid/diff ===")
mid_fwd, diff_fwd = traj_to_mid_diff(traj_fwd_train, "forward train")

print("\n=== Training forward memory-safe 3D EGNN ===")
loss_fwd = train(net_fwd, opt_fwd, sched_fwd, mid_fwd, diff_fwd, time_fwd, "forward")

del mid_fwd, diff_fwd
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

print("\n=== Evaluating forward sigma(t) ===")
sigma_fwd = eval_sigma_from_raw(net_fwd, traj_fwd_test, time_fwd, "forward test")

if device.type == "cuda":
    torch.cuda.empty_cache()

# ================================================================
# REVERSE
# ================================================================
print("\n=== Reconstructing reverse mid/diff ===")
mid_rev, diff_rev = traj_to_mid_diff(traj_rev_train, "reverse train")

print("\n=== Training reverse memory-safe 3D EGNN ===")
loss_rev = train(net_rev, opt_rev, sched_rev, mid_rev, diff_rev, time_rev, "reverse")

del mid_rev, diff_rev
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()

print("\n=== Evaluating reverse sigma(t) ===")
sigma_rev = eval_sigma_from_raw(net_rev, traj_rev_test, time_rev, "reverse test")

if device.type == "cuda":
    torch.cuda.empty_cache()

# ================================================================
# PLOT
# ================================================================
time_arr = np.arange(L, dtype=np.float32) * dt_inf
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(time_arr, sigma_fwd, lw=0.8, color="steelblue", label=r"$\sigma_\mathrm{fwd}(t)$")
axes[0, 0].set_title(f"Forward EPR (N_test={N_eval})")
axes[0, 0].set_xlabel("t [s]")
axes[0, 0].set_ylabel(r"$\sigma(t)$")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid()

axes[0, 1].plot(time_arr, sigma_rev[::-1], lw=0.8, color="tomato", label=r"$\sigma_\mathrm{rev}(t)$")
axes[0, 1].set_title("Reverse EPR - mapped to forward time")
axes[0, 1].set_xlabel("t [s]")
axes[0, 1].set_ylabel(r"$\sigma(t)$")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid()

axes[1, 0].plot(time_arr, sigma_fwd, lw=0.8, color="steelblue", label="forward")
axes[1, 0].plot(time_arr, sigma_rev[::-1], lw=0.8, color="tomato", label="reverse")
axes[1, 0].set_title("EPR - forward and reverse")
axes[1, 0].set_xlabel("t [s]")
axes[1, 0].set_ylabel(r"$\sigma(t)$")
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].semilogy(loss_fwd, color="steelblue", lw=0.7, label="forward")
axes[1, 1].semilogy(loss_rev, color="tomato", lw=0.7, label="reverse")
axes[1, 1].axvline(warmup_epochs, color="gray", lw=0.8, linestyle="--", label="warm-up end")
axes[1, 1].set_title("Training Loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend(fontsize=7)
axes[1, 1].grid()

t_plot = torch.linspace(0, t_max, 500, device=device)
with torch.inference_mode():
    phi_fwd = net_fwd.phi(t_plot).detach().cpu().numpy()
    phi_rev = net_rev.phi(t_plot).detach().cpu().numpy()

t_np = t_plot.detach().cpu().numpy()

axes[0, 2].set_title("Learned Gaussian bases - forward")
for k_idx in range(K):
    axes[0, 2].plot(t_np, phi_fwd[:, k_idx], lw=0.6, alpha=0.6)
axes[0, 2].set_xlabel("t [s]")
axes[0, 2].set_ylabel(r"$\phi_k(t)$")
axes[0, 2].grid()

axes[1, 2].set_title("Learned Gaussian bases - reverse")
for k_idx in range(K):
    axes[1, 2].plot(t_np, phi_rev[:, k_idx], lw=0.6, alpha=0.6)
axes[1, 2].set_xlabel("t [s]")
axes[1, 2].set_ylabel(r"$\phi_k(t)$")
axes[1, 2].grid()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "epr_fwd_rev_3d_egnn.png"), dpi=150)
plt.close()

# ================================================================
# SAVE RESULTS
# ================================================================
np.save(os.path.join(output_dir, "sigma_fwd.npy"), sigma_fwd)
np.save(os.path.join(output_dir, "sigma_rev.npy"), sigma_rev)
np.save(os.path.join(output_dir, "loss_fwd.npy"), np.array(loss_fwd, dtype=np.float32))
np.save(os.path.join(output_dir, "loss_rev.npy"), np.array(loss_rev, dtype=np.float32))
np.save(os.path.join(output_dir, "time_arr.npy"), time_arr)
np.save(os.path.join(output_dir, "dt_inf.npy"), np.array([dt_inf], dtype=np.float32))
np.save(os.path.join(output_dir, "N_train.npy"), np.array([N_train], dtype=np.int64))
np.save(os.path.join(output_dir, "N_test.npy"), np.array([N_eval], dtype=np.int64))

torch.save(net_fwd.state_dict(), os.path.join(output_dir, "net_fwd_3d_egnn.pt"))
torch.save(net_rev.state_dict(), os.path.join(output_dir, "net_rev_3d_egnn.pt"))

mu_fwd = net_fwd.mu.detach().cpu().numpy()
sig_fwd = net_fwd.log_sig.exp().detach().cpu().numpy()
mu_rev = net_rev.mu.detach().cpu().numpy()
sig_rev = net_rev.log_sig.exp().detach().cpu().numpy()

np.savez(
    os.path.join(output_dir, "gaussian_params_3d_egnn.npz"),
    mu_fwd=mu_fwd,
    sig_fwd=sig_fwd,
    mu_rev=mu_rev,
    sig_rev=sig_rev,
    N_train=np.array([N_train], dtype=np.int64),
    N_test=np.array([N_eval], dtype=np.int64),
    split_seed=np.array([seed], dtype=np.int64),
)

# ================================================================
# SUMMARY
# ================================================================
total_fwd = float(np.sum(sigma_fwd) * dt_inf)
total_rev = float(np.sum(sigma_rev) * dt_inf)

print(f"\nIndependent split: N_train={N_train}, N_test={N_eval}, seed={seed}")
print(f"Total EP forward   : {total_fwd:.4f} kB")
print(f"Total EP reverse   : {total_rev:.4f} kB")
print(f"Sum (should be >=0): {total_fwd + total_rev:.4f} kB")

print(f"\nForward Gaussian centres (mu_k):\n  {np.round(mu_fwd, 4)}")
print(f"Forward Gaussian widths  (sig_k):\n  {np.round(sig_fwd, 4)}")
print(f"\nResults saved to: {output_dir}")
print("Done.")
