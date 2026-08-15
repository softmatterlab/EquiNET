#!/usr/bin/env python3
import os
import sys
import math
import multiprocessing as mp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

# ============================================================
# 0) OUTPUT DIR + basic args
# ============================================================
if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python hairpin_3d_traj_only_fixed_y.py <OUTPUT_DIR> [N_total] [n_workers] [chunk_size]"
    )

OUTPUT_DIR = sys.argv[1]
N_total = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
N_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, mp.cpu_count() - 1)
CHUNK_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 100
BASE_SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 123456

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Parameters
# ================================================================
dt = 1e-5

n_steps_eq_xy_start = 150_000 #was 75000
n_steps_y_drive = 20_000      # kept as fixed-trap waiting stage
n_steps_x_drive = 600_000 #was 1200000
n_steps_eq_end = 150_000 #was 75000

gamma = 1.0
T = 1.0
kB = 1.0

n_beads = 13
dim = 3

k_bond = 500.0
k_bend = 20.0

k_anchor = 1000.0
k_pull = 10.0

v_x = 2.0 #was 1

epsilon_native = 3.0 #was 3
epsilon_rep = 1.0
sigma_rep = 0.8

store_stride = 5000

sqrt_2Tdt_over_gamma = np.sqrt(2.0 * kB * T * dt / gamma)

# Fixed trap start/end coordinate
TRAP_START = np.array([1.3548766, -1.8808337, -0.00921113], dtype=np.float64)

# ================================================================
# Native contacts
# ================================================================
native_pairs_1based = [(4, 7), (3, 8), (2, 9)]
native_pairs = [(i - 1, j - 1) for i, j in native_pairs_1based]
native_set = {tuple(sorted(p)) for p in native_pairs}

# ================================================================
# Initial folded hairpin guess
# ================================================================
def make_initial_hairpin_guess():
    pts2 = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [0.0, 4.0],
        [0.4, 4.8],
        [1.2, 4.2],
        [1.2, 3.2],
        [1.2, 2.2],
        [1.2, 1.2],
        [1.2, 0.2],
        [1.2, -0.8],
        [1.2, -1.8],
    ], dtype=np.float64)

    pts3 = np.zeros((n_beads, 3), dtype=np.float64)
    pts3[:, :2] = pts2
    return pts3

# ================================================================
# LJ helpers
# ================================================================
def U_LJ_scalar(r, epsilon, sigma):
    r = np.maximum(r, 1e-12)
    inv_r6 = (sigma / r) ** 6
    return 4.0 * epsilon * (inv_r6**2 - inv_r6)

def F_LJ_vec(dr, epsilon, sigma):
    r2 = np.sum(dr * dr, axis=-1, keepdims=True)
    r2 = np.maximum(r2, 1e-24)
    r = np.sqrt(r2)
    pref = 24.0 * epsilon * (
        2.0 * sigma**12 / r**14 - sigma**6 / r**8
    )
    return pref * dr

# ================================================================
# Single-configuration mechanics
# ================================================================
def F_bonds_single(r, bond_rest):
    F = np.zeros_like(r)
    dr = r[1:] - r[:-1]
    dist = np.linalg.norm(dr, axis=1, keepdims=True)
    dist = np.maximum(dist, 1e-12)
    rest = bond_rest[:, None]
    fpair = k_bond * (dist - rest) * dr / dist
    F[:-1] += fpair
    F[1:] -= fpair
    return F

def F_bend_single(r, b_ref):
    F = np.zeros_like(r)
    a = k_bend / b_ref**2
    c = r[2:] - 2.0 * r[1:-1] + r[:-2]

    for i in range(len(c)):
        ci = c[i]
        F[i] += -a * ci
        F[i + 1] += 2.0 * a * ci
        F[i + 2] += -a * ci

    return F

def F_native_contacts_single(r, native_sigma):
    F = np.zeros_like(r)

    for pair_idx, (i, j) in enumerate(native_pairs):
        dr = r[i] - r[j]
        fij = F_LJ_vec(dr[None, :], epsilon_native, native_sigma[pair_idx])[0]
        F[i] += fij
        F[j] -= fij

    return F

def F_repulsive_single(r):
    F = np.zeros_like(r)
    rcut = 2.0 ** (1.0 / 6.0) * sigma_rep

    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            if abs(i - j) == 1:
                continue
            if (i, j) in native_set:
                continue

            dr = r[i] - r[j]
            rij = np.linalg.norm(dr)

            if rij < rcut:
                fij = F_LJ_vec(dr[None, :], epsilon_rep, sigma_rep)[0]
                F[i] += fij
                F[j] -= fij

    return F

def F_anchor_single(r1, r1_trap):
    return -k_anchor * (r1 - r1_trap)

def F_pull_single(r_end, trap_end):
    return -k_pull * (r_end - trap_end)

def total_conservative_forces_single(
    r, bond_rest, b_ref, native_sigma, r1_trap, trap_end
):
    F = np.zeros_like(r)
    F += F_bonds_single(r, bond_rest)
    F += F_bend_single(r, b_ref)
    F += F_native_contacts_single(r, native_sigma)
    F += F_repulsive_single(r)
    F[0] += F_anchor_single(r[0], r1_trap)
    F[-1] += F_pull_single(r[-1], trap_end)
    return F

# ================================================================
# Batched mechanics
# ================================================================
def F_bonds_batch(R, bond_rest):
    F = np.zeros_like(R)
    dr = R[:, 1:] - R[:, :-1]
    dist = np.linalg.norm(dr, axis=2, keepdims=True)
    dist = np.maximum(dist, 1e-12)
    rest = bond_rest[None, :, None]
    fpair = k_bond * (dist - rest) * dr / dist
    F[:, :-1] += fpair
    F[:, 1:] -= fpair
    return F

def F_bend_batch(R, b_ref):
    F = np.zeros_like(R)
    a = k_bend / b_ref**2
    c = R[:, 2:] - 2.0 * R[:, 1:-1] + R[:, :-2]
    F[:, :-2] += -a * c
    F[:, 1:-1] += 2.0 * a * c
    F[:, 2:] += -a * c
    return F

def F_native_contacts_batch(R, native_sigma):
    F = np.zeros_like(R)

    for pair_idx, (i, j) in enumerate(native_pairs):
        dr = R[:, i] - R[:, j]
        fij = F_LJ_vec(dr, epsilon_native, native_sigma[pair_idx])
        F[:, i] += fij
        F[:, j] -= fij

    return F

def F_repulsive_batch(R):
    F = np.zeros_like(R)
    rcut = 2.0 ** (1.0 / 6.0) * sigma_rep

    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            if abs(i - j) == 1:
                continue
            if (i, j) in native_set:
                continue

            dr = R[:, i] - R[:, j]
            rij = np.linalg.norm(dr, axis=1)
            mask = rij < rcut

            if np.any(mask):
                fij = F_LJ_vec(dr[mask], epsilon_rep, sigma_rep)
                F[mask, i] += fij
                F[mask, j] -= fij

    return F

def F_anchor_batch(r1, r1_trap):
    return -k_anchor * (r1 - r1_trap[None, :])

def F_pull_batch(r_end, trap_end):
    return -k_pull * (r_end - trap_end[None, :])

def total_conservative_forces_batch(
    R, bond_rest, b_ref, native_sigma, r1_trap, trap_end
):
    F = np.zeros_like(R)
    F += F_bonds_batch(R, bond_rest)
    F += F_bend_batch(R, b_ref)
    F += F_native_contacts_batch(R, native_sigma)
    F += F_repulsive_batch(R)
    F[:, 0] += F_anchor_batch(R[:, 0], r1_trap)
    F[:, -1] += F_pull_batch(R[:, -1], trap_end)
    return F

# ================================================================
# Mechanical relaxation
# ================================================================
def relax_guess_to_folded_shape(
    r_guess, max_steps=200000, dt_relax=2e-4, tol=1e-8
):
    r = r_guess.copy()

    bond_rest_tmp = np.linalg.norm(r_guess[1:] - r_guess[:-1], axis=1)
    b_ref_tmp = np.mean(bond_rest_tmp)

    native_r0_tmp = np.array([
        np.linalg.norm(r_guess[i] - r_guess[j]) for i, j in native_pairs
    ], dtype=np.float64)
    native_sigma_tmp = native_r0_tmp / (2.0 ** (1.0 / 6.0))

    r1_trap = r_guess[0].copy()
    trap_end = r_guess[-1].copy()

    for step in range(max_steps):
        F = total_conservative_forces_single(
            r,
            bond_rest_tmp,
            b_ref_tmp,
            native_sigma_tmp,
            r1_trap,
            trap_end,
        )

        max_force = np.max(np.linalg.norm(F, axis=1))

        if max_force < tol:
            print(f"Mechanical pre-relaxation converged in {step} steps.")
            return r

        r += (F / gamma) * dt_relax

        if step % 20000 == 0:
            print(f"Pre-relax step {step}, max|F| = {max_force:.3e}")

    print("Warning: pre-relaxation hit max_steps.")
    return r

def build_equilibrium_model_from_relaxed_shape(r_relaxed):
    bond_rest = np.linalg.norm(r_relaxed[1:] - r_relaxed[:-1], axis=1)
    b_ref = np.mean(bond_rest)

    native_r0 = np.array([
        np.linalg.norm(r_relaxed[i] - r_relaxed[j]) for i, j in native_pairs
    ], dtype=np.float64)
    native_sigma = native_r0 / (2.0 ** (1.0 / 6.0))

    r1_trap = r_relaxed[0].copy()
    end_trap_0 = r_relaxed[-1].copy()

    return bond_rest, b_ref, native_sigma, r1_trap, end_trap_0

# ================================================================
# Revised protocol:
# no y-drive, fixed y/z, x-only pulling
# ================================================================
def build_trap_protocol():
    x0, y0, z0 = TRAP_START

    x_stage1 = np.full(n_steps_eq_xy_start, x0)
    y_stage1 = np.full(n_steps_eq_xy_start, y0)
    z_stage1 = np.full(n_steps_eq_xy_start, z0)

    # This stage is now fixed, not y-driven
    x_stage2 = np.full(n_steps_y_drive, x0)
    y_stage2 = np.full(n_steps_y_drive, y0)
    z_stage2 = np.full(n_steps_y_drive, z0)

    # x-only pulling
    x_stage3 = x0 + np.linspace(
        0.0,
        v_x * dt * (n_steps_x_drive - 1),
        n_steps_x_drive,
    )
    y_stage3 = np.full(n_steps_x_drive, y0)
    z_stage3 = np.full(n_steps_x_drive, z0)

    x_stage4 = np.full(n_steps_eq_end, x_stage3[-1])
    y_stage4 = np.full(n_steps_eq_end, y0)
    z_stage4 = np.full(n_steps_eq_end, z0)

    x_protocol = np.concatenate([x_stage1, x_stage2, x_stage3, x_stage4])
    y_protocol = np.concatenate([y_stage1, y_stage2, y_stage3, y_stage4])
    z_protocol = np.concatenate([z_stage1, z_stage2, z_stage3, z_stage4])

    trap_fwd = np.column_stack(
        [x_protocol, y_protocol, z_protocol]
    ).astype(np.float64)

    trap_rev = trap_fwd[::-1].copy()

    return trap_fwd, trap_rev

# ================================================================
# Batched thermalization
# ================================================================
def thermalize_batch(
    R, bond_rest, b_ref, native_sigma, r1_trap, trap_end_fixed, n_steps_therm, rng
):
    for _ in range(n_steps_therm):
        F = total_conservative_forces_batch(
            R, bond_rest, b_ref, native_sigma, r1_trap, trap_end_fixed
        )
        noise = sqrt_2Tdt_over_gamma * rng.standard_normal(R.shape)
        R += (F / gamma) * dt + noise

    return R

# ================================================================
# Batched protocol simulation with storage
# ================================================================
def simulate_protocol_store_batch(
    R, trap_protocol, bond_rest, b_ref, native_sigma, r1_trap, rng
):
    M = R.shape[0]
    n_steps_total = len(trap_protocol)
    n_store = (n_steps_total - 1) // store_stride + 1

    traj = np.empty((M, n_store, n_beads, dim), dtype=np.float32)
    traj[:, 0] = R.astype(np.float32)

    store_idx = 1

    for t in range(1, n_steps_total):
        trap_old = trap_protocol[t - 1]

        F = total_conservative_forces_batch(
            R, bond_rest, b_ref, native_sigma, r1_trap, trap_old
        )
        noise = sqrt_2Tdt_over_gamma * rng.standard_normal(R.shape)
        R += (F / gamma) * dt + noise

        if t % store_stride == 0:
            traj[:, store_idx] = R.astype(np.float32)
            store_idx += 1

    return traj, R

# ================================================================
# Globals for workers
# ================================================================
G = {}

def init_worker(params):
    global G
    G = params

def worker_simulate_chunk(task):
    start, stop, seed = task
    M = stop - start

    rng = np.random.default_rng(seed)

    bond_rest = G["bond_rest"]
    b_ref = G["b_ref"]
    native_sigma = G["native_sigma"]
    r1_trap = G["r1_trap"]
    r_eq = G["r_eq"]
    trap_fwd = G["trap_fwd"]
    trap_rev = G["trap_rev"]
    traj_fwd_path = G["traj_fwd_path"]
    traj_rev_path = G["traj_rev_path"]
    shape = G["traj_shape"]

    R0 = np.tile(r_eq[None, :, :], (M, 1, 1)).astype(np.float64)

    R0 = thermalize_batch(
        R0,
        bond_rest,
        b_ref,
        native_sigma,
        r1_trap,
        trap_fwd[0],
        n_steps_eq_xy_start,
        rng,
    )

    traj_fwd_batch, Rf = simulate_protocol_store_batch(
        R0.copy(),
        trap_fwd,
        bond_rest,
        b_ref,
        native_sigma,
        r1_trap,
        rng,
    )

    traj_rev_batch, _ = simulate_protocol_store_batch(
        Rf.copy(),
        trap_rev,
        bond_rest,
        b_ref,
        native_sigma,
        r1_trap,
        rng,
    )

    mm_fwd = np.lib.format.open_memmap(
        traj_fwd_path, mode="r+", dtype=np.float32, shape=shape
    )
    mm_rev = np.lib.format.open_memmap(
        traj_rev_path, mode="r+", dtype=np.float32, shape=shape
    )

    mm_fwd[start:stop] = traj_fwd_batch
    mm_rev[start:stop] = traj_rev_batch

    mm_fwd.flush()
    mm_rev.flush()

    del mm_fwd, mm_rev, traj_fwd_batch, traj_rev_batch, R0, Rf

    print(f"Finished chunk {start}:{stop}")
    return start, stop

# ================================================================
# Main
# ================================================================
def main():
    print("Building initial 3D hairpin guess...")
    r_guess = make_initial_hairpin_guess()

    print("Relaxing rough guess to folded mechanical state...")
    r_relaxed = relax_guess_to_folded_shape(r_guess)

    print("Freezing equilibrium geometry into model parameters...")
    bond_rest, b_ref, native_sigma, r1_trap, end_trap_0 = (
        build_equilibrium_model_from_relaxed_shape(r_relaxed)
    )

    r_eq = r_relaxed.copy()

    print("Building fixed-y, x-only trap protocol...")
    trap_fwd, trap_rev = build_trap_protocol()

    print("Forward trap starts at:", trap_fwd[0])
    print("Forward trap ends at:  ", trap_fwd[-1])
    print("Reverse trap starts at:", trap_rev[0])
    print("Reverse trap ends at:  ", trap_rev[-1])

    n_steps_total = len(trap_fwd)
    n_store = (n_steps_total - 1) // store_stride + 1

    traj_shape = (N_total, n_store, n_beads, dim)

    bytes_per_file = np.prod(traj_shape) * np.dtype(np.float32).itemsize

    print(f"N_total         = {N_total}")
    print(f"n_steps_total   = {n_steps_total}")
    print(f"store_stride    = {store_stride}")
    print(f"n_store         = {n_store}")
    print(f"traj shape      = {traj_shape}")
    print(f"Each file size  ≈ {bytes_per_file / 1e9:.2f} GB")
    print(f"Total output    ≈ {2 * bytes_per_file / 1e9:.2f} GB")

    traj_fwd_path = os.path.join(OUTPUT_DIR, "traj_fwd.npy")
    traj_rev_path = os.path.join(OUTPUT_DIR, "traj_rev.npy")

    print("Creating output memmaps...")
    np.lib.format.open_memmap(
        traj_fwd_path, mode="w+", dtype=np.float32, shape=traj_shape
    )
    np.lib.format.open_memmap(
        traj_rev_path, mode="w+", dtype=np.float32, shape=traj_shape
    )

    # Optional: save protocols too
    np.save(os.path.join(OUTPUT_DIR, "trap_fwd.npy"), trap_fwd)
    np.save(os.path.join(OUTPUT_DIR, "trap_rev.npy"), trap_rev)

    params = {
        "bond_rest": bond_rest,
        "b_ref": b_ref,
        "native_sigma": native_sigma,
        "r1_trap": r1_trap,
        "r_eq": r_eq,
        "trap_fwd": trap_fwd,
        "trap_rev": trap_rev,
        "traj_fwd_path": traj_fwd_path,
        "traj_rev_path": traj_rev_path,
        "traj_shape": traj_shape,
    }

    tasks = []
    seed_seq = np.random.SeedSequence(BASE_SEED)
    child_seeds = seed_seq.spawn(math.ceil(N_total / CHUNK_SIZE))

    chunk_id = 0
    for start in range(0, N_total, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N_total)
        seed = int(child_seeds[chunk_id].generate_state(1)[0])
        tasks.append((start, stop, seed))
        chunk_id += 1

    print(f"Running with {N_WORKERS} workers, chunk size {CHUNK_SIZE}...")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(params,),
    ) as pool:
        for _ in pool.imap_unordered(worker_simulate_chunk, tasks):
            pass

    print("\nDone. Saved:")
    print("  traj_fwd.npy")
    print("  traj_rev.npy")
    print("  trap_fwd.npy")
    print("  trap_rev.npy")

if __name__ == "__main__":
    main()
