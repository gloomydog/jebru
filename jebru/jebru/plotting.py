"""Publication-style static plot (matplotlib) used by the CLI and the GUI export."""
from __future__ import annotations

import numpy as np

from .processing import Processed


def plot_spectrum(pr: Processed, regions=None, peaks=None, path=None, xlim=None,
                  title=None, figsize=(11, 4.5), label_peaks=None):
    import matplotlib
    if path:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    regions = regions or []
    peaks = peaks or []
    x, y = pr.ppm, pr.real
    nuc = pr.spec.nucleus
    if xlim is None:
        if nuc == "1H":
            xlim = (max(10.0, _sig_max(pr) + 0.5), min(-0.5, _sig_min(pr) - 0.3))
        elif nuc == "13C":
            xlim = (max(200.0, _sig_max(pr) + 5), min(-5.0, _sig_min(pr) - 5))
        else:
            xlim = (x[0], x[-1])
    m = (x <= xlim[0]) & (x >= xlim[1])
    top = np.max(y[m]) if m.any() else y.max()
    if nuc == "13C":
        # solvent dominates: scale to the 2nd group of strong peaks
        hs = sorted([p.height for p in peaks], reverse=True)
        if hs:
            top = hs[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x[m], y[m], color="k", lw=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.22 * top, 1.18 * top)
    ax.set_xlabel(f"δ ({nuc}) / ppm")
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    # integrals
    span = abs(x[1] - x[0])
    last_x, level = None, 0
    width = abs(xlim[0] - xlim[1])
    for r in sorted(regions, key=lambda r: -r.start):
        if not r.include or r.nH < 0.5:
            continue
        sel = (x <= r.start) & (x >= r.stop)
        if sel.sum() < 2:
            continue
        cum = np.cumsum(y[sel][::-1])[::-1] * span
        if cum[0] != 0:
            cum = cum / abs(cum[0])
        ax.plot(x[sel], 0.92 * top + 0.12 * top * (1 - cum) - 0.06 * top, color="tab:red", lw=0.8)
        xm = (r.start + r.stop) / 2
        ax.annotate("", xy=(r.start, -0.07 * top), xytext=(r.stop, -0.07 * top),
                    arrowprops=dict(arrowstyle="-", color="tab:blue", lw=1.2))
        level = (level + 1) % 2 if last_x is not None and abs(last_x - xm) < 0.035 * width else 0
        last_x = xm
        ax.text(xm, -0.115 * top - level * 0.06 * top, f"{r.nH:.2f}", ha="center", va="top",
                fontsize=7, color="tab:blue")
    # peak labels
    if label_peaks is None:
        label_peaks = nuc != "1H"
    if label_peaks:
        for p in peaks:
            if xlim[1] <= p.ppm <= xlim[0]:
                ax.text(p.ppm, min(p.height, top) + 0.02 * top, f"{p.ppm:.1f}" if nuc == "13C" else f"{p.ppm:.2f}",
                        rotation=90, ha="center", va="bottom", fontsize=6)
    ax.set_title(title or pr.spec.name, fontsize=9, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig


def _sig_max(pr):
    from .analysis import pick_peaks
    pk = pick_peaks(pr, snr=20)
    return max((p.ppm for p in pk), default=pr.ppm[0])


def _sig_min(pr):
    from .analysis import pick_peaks
    pk = pick_peaks(pr, snr=20)
    return min((p.ppm for p in pk), default=pr.ppm[-1])
