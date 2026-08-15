#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent

DATA_FILE = HERE / "fig_3.npz"
OUTPUT_FILE = HERE / "fig_3.pdf"


# ============================================================
# Load previously saved figure data
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find figure data:\n{DATA_FILE}\n\n"
        "Run the data-generation script first to create "
        "'fig_work_deltaF_data.npz'."
    )

data = np.load(DATA_FILE)

print(f"Loaded figure data from: {DATA_FILE}")

# Main x-axis values
eps_vals = data["eps_vals"]

# Work-distribution crossing points
crossing_values = data["crossing_values"]

# Full-system free-energy estimates
delta_fwd_full = data["delta_fwd_full"]
delta_rev_full = data["delta_rev_full"]

# Coarse-grained free-energy estimates
delta_fwd_cg = data["delta_fwd_cg"]
delta_rev_cg = data["delta_rev_cg"]

# Corresponding uncertainties
delta_fwd_full_err = data["delta_fwd_full_err"]
delta_rev_full_err = data["delta_rev_full_err"]

delta_fwd_cg_err = data["delta_fwd_cg_err"]
delta_rev_cg_err = data["delta_rev_cg_err"]


# ============================================================
# Style
# ============================================================

sns.set_style("white")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.2,
    "legend.handlelength": 1.5,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
    "mathtext.default": "it",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 200,
    "savefig.dpi": 600,
})

COLOR_FWD = "#0072B2"
COLOR_REV = "#D55E00"


# ============================================================
# Linear fit
# ============================================================

fit_coeffs = np.polyfit(
    eps_vals,
    delta_fwd_full,
    deg=1,
)

fit_line = np.poly1d(fit_coeffs)

x_fit = np.linspace(
    2.5,
    12.5,
    300,
)


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(
    figsize=(7.2, 1.8)
)

gs = fig.add_gridspec(
    1,
    6,
    width_ratios=[
        1.0,
        1.0,
        1.0,
        1.0,
        0.10,
        1.15,
    ],
    wspace=0.12,
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3]),
]

ax_df = fig.add_subplot(gs[0, 5])


# ============================================================
# Work-distribution panels
# ============================================================

for i, (ax, eps, w_cross) in enumerate(
    zip(
        axes,
        eps_vals,
        crossing_values,
    )
):

    key = f"eps{int(eps)}"

    # --------------------------------------------------------
    # Load previously saved work samples
    # --------------------------------------------------------

    wf = data[f"{key}_work_f"]
    wr = data[f"{key}_minus_work_r"]

    wf_v0p2 = data[
        f"{key}_work_f_v0p2"
    ]

    wr_v0p2 = data[
        f"{key}_minus_work_r_v0p2"
    ]

    # --------------------------------------------------------
    # Main distributions
    # --------------------------------------------------------

    sns.histplot(
        wf,
        bins=15,
        stat="density",
        color=COLOR_FWD,
        alpha=0.35,
        kde=True,
        line_kws={"lw": 1.6},
        ax=ax,
    )

    sns.histplot(
        wr,
        bins=15,
        stat="density",
        color=COLOR_REV,
        alpha=0.35,
        kde=True,
        line_kws={"lw": 1.6},
        ax=ax,
    )

    # --------------------------------------------------------
    # v = 0.2 distributions
    # --------------------------------------------------------

    sns.histplot(
        wf_v0p2,
        bins=15,
        stat="density",
        color=COLOR_FWD,
        alpha=0.60,
        kde=False,
        ax=ax,
    )

    sns.histplot(
        wr_v0p2,
        bins=20,
        stat="density",
        color=COLOR_REV,
        alpha=0.60,
        kde=False,
        ax=ax,
    )

    # --------------------------------------------------------
    # Crossing point
    # --------------------------------------------------------

    ax.axvline(
        w_cross,
        color="black",
        linestyle="-",
        linewidth=1.0,
        alpha=0.35,
        zorder=5,
    )

    # --------------------------------------------------------
    # Full-system Delta F
    # --------------------------------------------------------

    ax.axvline(
        delta_fwd_full[i],
        color=COLOR_FWD,
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )

    ax.axvspan(
        delta_fwd_full[i]
        - delta_fwd_full_err[i],
        delta_fwd_full[i]
        + delta_fwd_full_err[i],
        color=COLOR_FWD,
        alpha=0.10,
        linewidth=0,
    )

    # --------------------------------------------------------
    # Coarse-grained Delta F
    # --------------------------------------------------------

    ax.axvline(
        delta_fwd_cg[i],
        color=COLOR_FWD,
        linestyle=":",
        linewidth=1.0,
        zorder=4,
    )

    ax.axvspan(
        delta_fwd_cg[i]
        - delta_fwd_cg_err[i],
        delta_fwd_cg[i]
        + delta_fwd_cg_err[i],
        color=COLOR_FWD,
        alpha=0.05,
        linewidth=0,
    )

    # --------------------------------------------------------
    # Epsilon label
    # --------------------------------------------------------

    ax.text(
        0.00,
        1.12,
        rf"$\epsilon/k_BT={int(eps)}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="darkgreen",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            boxstyle="square,pad=0.08",
            alpha=0.80,
        ),
    )

    # --------------------------------------------------------
    # Axes formatting
    # --------------------------------------------------------

    ax.set_xlim(
        -150,
        250,
    )

    ax.set_xlabel(
        r"$W/k_BT$",
        labelpad=1.5,
    )

    if i == 0:
        ax.set_ylabel(
            "Density",
            labelpad=2,
        )
    else:
        ax.set_ylabel("")
        ax.tick_params(
            labelleft=False,
        )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator(2)
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    ax.grid(False)


# ============================================================
# Figure-level legend for work distributions
# ============================================================

hist_handles = [
    Line2D(
        [0], [0],
        color=COLOR_FWD,
        lw=1.5,
        label=r"$P_{\rm F}(W)$",
    ),

    Line2D(
        [0], [0],
        color=COLOR_REV,
        lw=1.5,
        label=r"$P_{\rm R}(-W)$",
    ),

    Line2D(
        [0], [0],
        color=COLOR_FWD,
        lw=1.2,
        linestyle="--",
        label=r"$\widehat{\Delta F}^{\,f}$",
    ),

    Line2D(
        [0], [0],
        color=COLOR_FWD,
        lw=1.1,
        linestyle=":",
        label=r"$\widehat{\Delta F}^{\,f}_{x}$",
    ),

    Line2D(
        [0], [0],
        color="black",
        lw=1.0,
        alpha=0.5,
        label=r"$W^\ast$",
    ),
]

fig.legend(
    handles=hist_handles,
    loc="upper center",
    bbox_to_anchor=(0.39, 1.04),
    ncol=5,
    frameon=False,
    fontsize=6.0,
    handlelength=1.4,
    columnspacing=1.2,
)


# ============================================================
# Delta F vs epsilon panel
# ============================================================

ax_df.errorbar(
    eps_vals,
    delta_fwd_full,
    yerr=delta_fwd_full_err,
    fmt="o",
    ms=4.5,
    mew=0.7,
    mec="white",
    capsize=2.0,
    capthick=0.9,
    elinewidth=0.9,
    color=COLOR_FWD,
    label=r"$\widehat{\Delta F}^{\rm f}$",
)

ax_df.plot(
    eps_vals,
    crossing_values,
    "D",
    ms=2.6,
    mew=0.2,
    mec="black",
    mfc="red",
    color="red",
    linestyle="none",
    zorder=5,
    label=r"$W^\ast$",
)

ax_df.errorbar(
    eps_vals,
    delta_fwd_cg,
    yerr=delta_fwd_cg_err,
    fmt="o",
    ms=3.8,
    mew=0.8,
    mec=COLOR_FWD,
    mfc="white",
    capsize=2.0,
    capthick=0.9,
    elinewidth=0.9,
    color=COLOR_FWD,
    linestyle="none",
    label=r"$\widehat{\Delta F}^{\rm f}_{x}$",
)

ax_df.errorbar(
    eps_vals,
    delta_rev_full,
    yerr=delta_rev_full_err,
    fmt="s",
    ms=3.8,
    mew=0.7,
    mec="white",
    capsize=2.0,
    capthick=0.9,
    elinewidth=0.9,
    color=COLOR_REV,
    label=r"$\widehat{\Delta F}^{\rm r}$",
)

ax_df.errorbar(
    eps_vals,
    delta_rev_cg,
    yerr=delta_rev_cg_err,
    fmt="s",
    ms=3.8,
    mew=0.8,
    mec=COLOR_REV,
    mfc="white",
    capsize=2.0,
    capthick=0.9,
    elinewidth=0.9,
    color=COLOR_REV,
    linestyle="none",
    label=r"$\widehat{\Delta F}^{\rm r}_{x}$",
)

ax_df.plot(
    x_fit,
    fit_line(x_fit),
    "--",
    color="0.25",
    lw=1.0,
    zorder=0,
    label="Linear fit",
)


# ============================================================
# Delta F panel formatting
# ============================================================

ax_df.set_xlim(
    1.5,
    13,
)

ax_df.set_xticks(
    [3, 6, 9, 12]
)

ax_df.set_xlabel(
    r"$\epsilon/k_BT$",
    labelpad=1.5,
)

ax_df.set_ylabel(
    r"$\widehat{\Delta F}/k_BT$",
    labelpad=2,
)

ax_df.xaxis.set_minor_locator(
    AutoMinorLocator(2)
)

ax_df.yaxis.set_minor_locator(
    AutoMinorLocator(2)
)

for spine in [
    "top",
    "right",
    "bottom",
    "left",
]:
    ax_df.spines[spine].set_linewidth(
        0.8
    )

ax_df.grid(False)

ax_df.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=3,
    fontsize=5.5,
    frameon=False,
    handlelength=1.2,
    columnspacing=0.9,
    handletextpad=0.4,
    borderaxespad=0.0,
)


# ============================================================
# Final layout
# ============================================================

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    bottom=0.24,
    top=0.83,
)


# ============================================================
# Tick formatting
# ============================================================

for ax in list(axes) + [ax_df]:

    ax.minorticks_on()

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2,
        width=0.6,
        bottom=True,
        left=True,
        top=False,
        right=False,
        color="black",
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=1,
        width=0.4,
        bottom=True,
        left=True,
        top=False,
        right=False,
        color="black",
    )


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_FILE,
    bbox_inches="tight",
    transparent=True,
    pad_inches=0.02,
)

plt.close(fig)

print(f"Saved figure to: {OUTPUT_FILE}")
