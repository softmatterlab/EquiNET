#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    AutoMinorLocator,
    LogLocator,
    NullFormatter,
)


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent

DATA_FILE = (
    HERE
    / "fig4.npz"
)

OUTPUT_PDF = (
    HERE
    / "fig4.pdf"
)

OUTPUT_PNG = (
    HERE
    / "fig4.png"
)


# ============================================================
# Load figure data
# ============================================================

if not DATA_FILE.is_file():

    raise FileNotFoundError(
        f"Could not find:\n"
        f"{DATA_FILE}\n\n"
        f"Run make_fig4_data.py first."
    )


data = np.load(
    DATA_FILE
)

print(
    f"Loaded: {DATA_FILE}"
)


# ============================================================
# Configuration
# ============================================================

EPSILONS = [
    int(x)
    for x in data[
        "epsilons"
    ]
]

DATA_SIZES = [
    int(x)
    for x in data[
        "data_sizes"
    ]
]

T_EQ1_END = float(
    data[
        "t_eq1_end"
    ][0]
)

T_DRIVE_END = float(
    data[
        "t_drive_end"
    ][0]
)


PANEL_LABELS = {
    3: r"$\epsilon/k_{\rm B}T=3$",
    6: r"$\epsilon/k_{\rm B}T=6$",
    9: r"$\epsilon/k_{\rm B}T=9$",
    12: r"$\epsilon/k_{\rm B}T=12$",
}


# ============================================================
# Style
# ============================================================

sns.set_style(
    "white"
)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.handlelength": 1.6,
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
# N-dependent colors for rows 1 and 2
# ============================================================

palette = sns.color_palette(
    "colorblind"
)

N_COLORS = {
    100: palette[2],
    500: palette[3],
    5000: palette[4],
    50000: palette[5],
}


# ============================================================
# Figure
#
# Row 0: forward sigma(t) Delta t
# Row 1: reverse sigma(t) Delta t
# Row 2: normalized total EP convergence
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(7.2, 5.5),
    sharex=False,
)


# ============================================================
# Protocol-region shading for upper 8 panels
# ============================================================

for ax in axes[:2, :].ravel():

    # Initial equilibration
    ax.axvspan(
        0.0,
        T_EQ1_END,
        color="0.7",
        alpha=0.072,
        linewidth=0,
        zorder=0,
    )

    # Driven interval
    ax.axvspan(
        T_EQ1_END,
        T_DRIVE_END,
        color="gold",
        alpha=0.06,
        linewidth=0,
        zorder=0,
    )

    # Final equilibration
    ax.axvspan(
        T_DRIVE_END,
        1.0,
        color="0.7",
        alpha=0.072,
        linewidth=0,
        zorder=0,
    )

    # Protocol boundaries
    ax.axvline(
        T_EQ1_END,
        color="0.35",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
        zorder=1,
    )

    ax.axvline(
        T_DRIVE_END,
        color="0.35",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
        zorder=1,
    )


# ============================================================
# Rows 0 and 1
#
# Forward and reverse entropy-production rates
# ============================================================

for col, epsilon in enumerate(
    EPSILONS
):

    ax_fwd = axes[
        0,
        col,
    ]

    ax_rev = axes[
        1,
        col,
    ]

    for n_data in DATA_SIZES:

        prefix = (
            f"eps{epsilon}_"
            f"N{n_data}"
        )

        required = [
            f"{prefix}_time",
            f"{prefix}_fwd_mean",
            f"{prefix}_fwd_std",
            f"{prefix}_rev_mean",
            f"{prefix}_rev_std",
        ]

        if not all(
            key in data.files
            for key in required
        ):
            continue

        time = data[
            f"{prefix}_time"
        ]

        fwd_mean = data[
            f"{prefix}_fwd_mean"
        ]

        fwd_std = data[
            f"{prefix}_fwd_std"
        ]

        rev_mean = data[
            f"{prefix}_rev_mean"
        ]

        rev_std = data[
            f"{prefix}_rev_std"
        ]

        color = N_COLORS[
            n_data
        ]

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        ax_fwd.plot(
            time,
            fwd_mean,
            color=color,
            linewidth=0.9,
            zorder=3,
        )

        ax_fwd.fill_between(
            time,
            fwd_mean - fwd_std,
            fwd_mean + fwd_std,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

        # ----------------------------------------------------
        # Reverse
        # ----------------------------------------------------

        ax_rev.plot(
            time,
            rev_mean,
            color=color,
            linewidth=0.9,
            zorder=3,
        )

        ax_rev.fill_between(
            time,
            rev_mean - rev_std,
            rev_mean + rev_std,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    # --------------------------------------------------------
    # Epsilon labels
    # --------------------------------------------------------

    ax_fwd.text(
        0.95,
        0.92,
        PANEL_LABELS[
            epsilon
        ],
        transform=ax_fwd.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="darkgreen",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.10",
            "alpha": 0.9,
        },
    )


# ============================================================
# Upper-panel formatting
# ============================================================

for row in [
    0,
    1,
]:

    for col in range(4):

        ax = axes[
            row,
            col,
        ]

        ax.set_xlim(
            0,
            1,
        )

        ax.set_xticks(
            np.linspace(
                0,
                1,
                6,
            )
        )

        ax.axhline(
            0,
            color="0.25",
            linestyle="--",
            linewidth=0.55,
            alpha=0.35,
            zorder=1,
        )

        ax.xaxis.set_minor_locator(
            AutoMinorLocator(2)
        )

        ax.yaxis.set_minor_locator(
            AutoMinorLocator(2)
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.spines[
            "bottom"
        ].set_linewidth(
            0.8
        )

        ax.spines[
            "left"
        ].set_linewidth(
            0.8
        )

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=3,
            width=0.7,
            top=False,
            right=False,
        )

        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=1.6,
            width=0.5,
            top=False,
            right=False,
        )

        ax.grid(False)


# ============================================================
# Labels for upper rows
# ============================================================

axes[
    0,
    0,
].set_ylabel(
    r"Forward $\sigma(t)\Delta t$"
)

axes[
    1,
    0,
].set_ylabel(
    r"Reverse $\sigma(t)\Delta t$"
)


for ax in axes[
    1,
    :
]:

    ax.set_xlabel(
        r"$t/\tau$"
    )


# ============================================================
# Bottom row: normalized total entropy production
# ============================================================

for col, epsilon in enumerate(
    EPSILONS
):

    ax = axes[
        2,
        col,
    ]

    prefix = (
        f"conv_eps{epsilon}"
    )

    n_values = np.asarray(
        data[
            f"{prefix}_N"
        ],
        dtype=float,
    )

    fwd_mean = np.asarray(
        data[
            f"{prefix}_fwd_mean"
        ],
        dtype=float,
    )

    fwd_std = np.asarray(
        data[
            f"{prefix}_fwd_std"
        ],
        dtype=float,
    )

    rev_mean = np.asarray(
        data[
            f"{prefix}_rev_mean"
        ],
        dtype=float,
    )

    rev_std = np.asarray(
        data[
            f"{prefix}_rev_std"
        ],
        dtype=float,
    )

    order = np.argsort(
        n_values
    )

    n_values = n_values[
        order
    ]

    fwd_mean = fwd_mean[
        order
    ]

    fwd_std = fwd_std[
        order
    ]

    rev_mean = rev_mean[
        order
    ]

    rev_std = rev_std[
        order
    ]


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    ax.errorbar(
        n_values,
        fwd_mean,
        yerr=fwd_std,
        color=COLOR_FWD,
        marker="o",
        markersize=3.2,
        markeredgewidth=0.5,
        markeredgecolor="white",
        linewidth=1.0,
        capsize=1.5,
        capthick=0.7,
        elinewidth=0.7,
        zorder=3,
    )


    # --------------------------------------------------------
    # Reverse
    # --------------------------------------------------------

    ax.errorbar(
        n_values,
        rev_mean,
        yerr=rev_std,
        color=COLOR_REV,
        marker="s",
        markersize=3.0,
        markeredgewidth=0.5,
        markeredgecolor="white",
        linewidth=1.0,
        capsize=1.5,
        capthick=0.7,
        elinewidth=0.7,
        alpha=0.65,
        zorder=3,
    )


    # --------------------------------------------------------
    # Reference = unity
    # --------------------------------------------------------

    ax.axhline(
        1.0,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
        zorder=1,
    )


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        80,
        max(
            7e4,
            1.25 * n_values.max(),
        ),
    )

    ax.set_ylim(
        0.5,
        1.2,
    )

    # Use actual convergence N values as major ticks.
    ax.set_xticks(
        n_values
    )

    labels = []

    for n in n_values:

        n_int = int(n)

        if n_int == 100:
            labels.append(
                "100"
            )

        elif n_int == 500:
            labels.append(
                "500"
            )

        elif n_int == 1000:
            labels.append(
                r"$10^3$"
            )

        elif n_int == 5000:
            labels.append(
                r"$5{\times}10^3$"
            )

        elif n_int == 50000:
            labels.append(
                r"$5{\times}10^4$"
            )

        elif n_int == 100000:
            labels.append(
                r"$10^5$"
            )

        else:
            labels.append(
                f"{n_int:g}"
            )

    ax.set_xticklabels(
        labels
    )

    ax.xaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=np.arange(
                2,
                10,
            ) * 0.1,
        )
    )

    ax.xaxis.set_minor_formatter(
        NullFormatter()
    )

    ax.text(
        0.95,
        0.92,
        PANEL_LABELS[
            epsilon
        ],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="darkgreen",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "boxstyle": "square,pad=0.10",
            "alpha": 0.9,
        },
    )

    ax.set_xlabel(
        r"Number of trajectories, $N$"
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        0.8
    )

    ax.spines[
        "left"
    ].set_linewidth(
        0.8
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3,
        width=0.7,
        top=False,
        right=False,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=1.6,
        width=0.5,
        top=False,
        right=False,
    )

    ax.grid(
        False
    )


axes[
    2,
    0,
].set_ylabel(
    r"Normalized "
    r"$\langle\Delta S_{\rm tot}\rangle$"
)


# ============================================================
# Hide repeated y tick labels
# ============================================================

for row in range(3):

    for col in range(
        1,
        4,
    ):

        axes[
            row,
            col,
        ].tick_params(
            labelleft=False
        )


# ============================================================
# N legend for upper panels
# ============================================================

def format_n(n):

    if n == 5000:

        return (
            r"$N=5\times10^3$"
        )

    if n == 50000:

        return (
            r"$N=5\times10^4$"
        )

    return (
        rf"$N={n}$"
    )


n_handles = [

    Line2D(
        [0],
        [0],
        color=N_COLORS[
            n_data
        ],
        linewidth=1.3,
        label=format_n(
            n_data
        ),
    )

    for n_data in DATA_SIZES
]


region_handles = [

    Patch(
        facecolor="0.7",
        alpha=0.20,
        edgecolor="none",
        label="Equilibration",
    ),

    Patch(
        facecolor="gold",
        alpha=0.15,
        edgecolor="none",
        label="Driving",
    ),

]


fig.legend(
    handles=(
        n_handles
        + region_handles
    ),
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        0.995,
    ),
    ncol=6,
    frameon=False,
    fontsize=7,
)


# ============================================================
# Bottom-row legend
# ============================================================

bottom_handles = [

    Line2D(
        [0],
        [0],
        color=COLOR_FWD,
        marker="o",
        markersize=4,
        markeredgecolor="white",
        linewidth=1.1,
        label="Forward",
    ),

    Line2D(
        [0],
        [0],
        color=COLOR_REV,
        marker="s",
        markersize=4,
        markeredgecolor="white",
        linewidth=1.1,
        label="Reverse",
    ),

    Line2D(
        [0],
        [0],
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        label=r"$N=5\times10^4$ reference",
    ),

]


fig.legend(
    handles=bottom_handles,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        0.352,
    ),
    ncol=3,
    frameon=False,
    fontsize=7,
)


# ============================================================
# Panel letters
# ============================================================

letters = [
    "(a)", "(b)", "(c)", "(d)",
    "(e)", "(f)", "(g)", "(h)",
    "(i)", "(j)", "(k)", "(l)",
]


for letter, ax in zip(
    letters,
    axes.ravel(),
):

    ax.text(
        -0.17,
        1.03,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.995,
    bottom=0.085,
    top=0.925,
    wspace=0.14,
    hspace=0.55,
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    transparent=True,
    pad_inches=0.03,
)

fig.savefig(
    OUTPUT_PNG,
    bbox_inches="tight",
    transparent=False,
    pad_inches=0.03,
    dpi=600,
)

plt.close(
    fig
)

print(
    f"Saved PDF: {OUTPUT_PDF}"
)

print(
    f"Saved PNG: {OUTPUT_PNG}"
)
