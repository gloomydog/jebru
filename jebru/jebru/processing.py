"""1D processing: digital filter, apodization, zero filling, FT, phase, baseline, referencing."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import optimize, ndimage

from .io import Spectrum1D

# (1H ppm, 13C ppm) residual solvent signals (Fulmer et al., Organometallics 2010,
# 29, 2176, Tables 1/2; pyridine-d5 not in that table)
SOLVENTS = {
    "CDCl3": (7.26, 77.16),
    "DMSO-d6": (2.50, 39.52),
    "CD3OD": (3.31, 49.00),
    "D2O": (4.79, None),
    "Acetone-d6": (2.05, 29.84),
    "C6D6": (7.16, 128.06),
    "CD3CN": (1.94, 1.32),
    "CD2Cl2": (5.32, 53.84),
    "THF-d8": (3.58, 67.21),
    "Toluene-d8": (2.08, 20.43),
    "C6D5Cl": (7.14, 134.19),
    "TFE-d3": (3.88, 61.50),
    "Pyridine-d5": (8.74, 150.35),
    "TMS": (0.00, 0.00),
}
_SOLV_ALIASES = {
    "CDCL3": "CDCl3", "CHLOROFORM-D": "CDCl3", "CHLOROFORM": "CDCl3",
    "DMSO": "DMSO-d6", "DMSO-D6": "DMSO-d6", "DMSO-D": "DMSO-d6",
    "MEOD": "CD3OD", "CD3OD": "CD3OD", "METHANOL-D4": "CD3OD", "METHANOL": "CD3OD",
    "D2O": "D2O", "H2O+D2O": "D2O", "DEUTERIUM OXIDE": "D2O",
    "ACETONE": "Acetone-d6", "ACETONE-D6": "Acetone-d6",
    "C6D6": "C6D6", "BENZENE-D6": "C6D6", "BENZENE": "C6D6",
    "CD3CN": "CD3CN", "ACETONITRILE-D3": "CD3CN", "ACN": "CD3CN",
    "CD2CL2": "CD2Cl2", "DICHLOROMETHANE-D2": "CD2Cl2",
    "THF": "THF-d8", "THF-D8": "THF-d8", "TOLUENE-D8": "Toluene-d8", "TOL": "Toluene-d8",
    "PYR": "Pyridine-d5", "PYRIDINE-D5": "Pyridine-d5",
    "C6D5CL": "C6D5Cl", "CHLOROBENZENE-D5": "C6D5Cl", "CHLOROBENZENE": "C6D5Cl",
    "TFE-D3": "TFE-d3", "TFE": "TFE-d3", "TRIFLUOROETHANOL-D3": "TFE-d3", "CF3CD2OD": "TFE-d3",
}


def norm_solvent(s: str) -> str:
    s = (s or "").strip()
    return _SOLV_ALIASES.get(s.upper(), s if s in SOLVENTS else "")


@dataclass
class ProcParams:
    lb: float = -1.0            # exponential line broadening [Hz]; <0 -> nucleus default
    zf: int = 2                 # zero filling factor (size -> next pow2 * zf)
    p0: float = 0.0
    p1: float = 0.0
    pivot_ppm: float | None = None
    auto_phase: bool = True
    baseline: bool = True
    baseline_order: int = 4
    baseline_method: str = "spline"   # spline | poly
    ref_mode: str = "auto"      # auto | none | manual
    solvent: str = ""           # override solvent for referencing
    ref_shift: float = 0.0      # manual: ppm offset added to axis
    # manual baseline (overrides the automatic one when points/regions exist).
    # Positions are in ppm on the *unreferenced* axis so they follow re-referencing.
    bl_points: list = field(default_factory=list)
    bl_regions: list = field(default_factory=list)      # [(high_ppm, low_ppm), ...]
    bl_manual_method: str = "spline"                     # spline (PCHIP) | poly
    bl_manual_order: int = 3

    def to_dict(self):
        return asdict(self)


@dataclass
class Processed:
    ppm: np.ndarray
    real: np.ndarray
    cplx: np.ndarray
    params: ProcParams
    spec: Spectrum1D
    ref_note: str = ""
    noise: float = 0.0
    log: list = field(default_factory=list)
    raw_real: np.ndarray | None = None      # real part before baseline correction
    baseline: np.ndarray | None = None      # subtracted baseline
    ref_applied: float = 0.0                # ppm shift added by referencing

    @property
    def hz_per_pt(self) -> float:
        return abs(self.ppm[0] - self.ppm[1]) * self.spec.obs

    def ppm_to_index(self, ppm):
        # ppm axis is decreasing with index
        return (self.ppm[0] - np.asarray(ppm)) / (self.ppm[0] - self.ppm[-1]) * (len(self.ppm) - 1)


# --------------------------------------------------------------- helpers
def remove_group_delay(fid: np.ndarray, gd: float) -> np.ndarray:
    """Remove digital-filter group delay (circular time shift by gd points)."""
    if gd <= 0:
        return fid
    n = fid.size
    spec = np.fft.fftshift(np.fft.fft(fid))
    k = np.arange(n) - n // 2
    spec = spec * np.exp(2j * np.pi * gd * k / n)
    out = np.fft.ifft(np.fft.ifftshift(spec))
    skip = int(np.floor(gd)) + 2
    return out[:-skip] if skip < n else out


def phase(data: np.ndarray, p0: float, p1: float, pivot: float = 0.0) -> np.ndarray:
    """p0, p1 in degrees; pivot as fraction (0..1) of the spectrum (index space)."""
    n = data.size
    x = np.arange(n) / n - pivot
    return data * np.exp(1j * np.deg2rad(p0 + p1 * x))


def _decimate_for_phase(spec: np.ndarray, target: int = 8192):
    step = max(1, spec.size // target)
    if step == 1:
        return spec, 1
    # keep peak heights roughly by taking the max-magnitude point in each block
    m = spec.size // step * step
    blk = spec[:m].reshape(-1, step)
    idx = np.argmax(np.abs(blk), axis=1)
    return blk[np.arange(blk.shape[0]), idx], step


def _peak_regions(mag: np.ndarray, thr: float, pad: int):
    m = mag > thr
    m = ndimage.binary_dilation(m, iterations=pad)
    lab, n = ndimage.label(m)
    return [np.where(lab == k)[0] for k in range(1, n + 1)]


def auto_phase(spec: np.ndarray, pivot: float, p1_max: float = 720.0):
    """Regional phase fit.

    For each isolated signal region the zero-order phase that makes the
    region's integral purely real/positive is -angle(sum(S)).  A circular
    weighted linear fit of those phases versus position gives p0/p1.
    A final ACME-entropy polish refines the result.
    """
    n = spec.size
    mag = np.abs(spec)
    noise = estimate_noise(np.real(spec))
    width = max(1, n // 32768)
    sm = ndimage.uniform_filter1d(mag, width) if width > 1 else mag
    med = np.median(sm)
    lo = max(med + 8 * 1.4826 * np.median(np.abs(sm - med)), 1e-4 * sm.max())
    regs = []
    for thr in np.geomspace(max(lo, 0.2 * sm.max()), lo, 12):
        rr = _peak_regions(sm, thr, pad=max(4, n // 8192))
        if len(rr) >= len(regs):
            regs = rr
    if not regs:
        return 0.0, 0.0
    xs, ph, w = [], [], []
    for r in regs:
        z = spec[r].sum()
        xs.append(r.mean() / n - pivot)
        ph.append(-np.angle(z))
        w.append(np.abs(z))
    xs, ph, w = map(np.asarray, (xs, ph, w))
    order = np.argsort(w)[::-1][:60]
    xs, ph, w = xs[order], ph[order], w[order] / w[order].max()
    w = np.sqrt(w)

    def circ_cost(p1r):
        p0r = np.angle(np.sum(w * np.exp(1j * (ph - p1r * xs))))
        c = np.sum(w * (1 - np.cos(ph - p0r - p1r * xs)))
        return c + 1e-3 * w.sum() * (p1r / (2 * np.pi)) ** 2, p0r

    if xs.size < 3 or np.ptp(xs) < 0.05:
        p0 = np.rad2deg(np.angle(np.sum(w * np.exp(1j * ph))))
        return float((p0 + 180) % 360 - 180), 0.0
    grid = np.deg2rad(np.linspace(-p1_max, p1_max, int(2 * p1_max) + 1))
    costs = [circ_cost(g)[0] for g in grid]
    g0 = grid[int(np.argmin(costs))]
    r = optimize.minimize_scalar(lambda v: circ_cost(v)[0], bounds=(g0 - 0.02, g0 + 0.02), method="bounded")
    p1r = r.x
    p0r = circ_cost(p1r)[1]
    p0, p1 = np.rad2deg(p0r), np.rad2deg(p1r)

    # ACME polish (Chen et al. 2002) on the full spectrum
    s = spec / (mag.max() or 1)
    x = np.arange(n) / n - pivot

    def acme(v):
        rr = np.real(s * np.exp(1j * np.deg2rad(v[0] + v[1] * x)))
        d = np.abs(np.diff(rr))
        h = d / (d.sum() + 1e-300)
        ent = -np.sum(h * np.log(h + 1e-300))
        return ent + 1000.0 * np.sum(np.minimum(rr, 0) ** 2)

    # several starts (regional fit, regional p0 with p1=0, and the p0 that is
    # best for p1=0); keep the lowest ACME cost
    # (only when S/N is high: on noisy spectra, e.g. 13C, the entropy measure
    # is dominated by noise and the regional fit is more reliable)
    snr = float(mag.max() / (estimate_noise(mag) + 1e-30))
    if snr > 2000:
        p0_flat = float(np.rad2deg(np.angle(np.sum(w * np.exp(1j * ph)))))
        best = None
        for start in ([p0, p1], [p0, 0.0], [p0_flat, 0.0]):
            r = optimize.minimize(acme, start, method="Nelder-Mead",
                                  options=dict(xatol=0.01, fatol=1e-10, maxiter=800))
            if best is None or r.fun < best.fun:
                best = r
        p0, p1 = best.x
    else:
        r = optimize.minimize(acme, [p0, p1], method="Nelder-Mead",
                              options=dict(xatol=0.01, fatol=1e-10, maxiter=600))
        if abs(r.x[0] - p0) < 30 and abs(r.x[1] - p1) < 60:
            p0, p1 = r.x
    return float((p0 + 180) % 360 - 180), float(p1)


def estimate_noise(y: np.ndarray) -> float:
    """Noise sigma: std of linearly detrended segments, low percentile
    (segments without signal).  Robust against zero-filling correlation."""
    y = np.asarray(y, float)
    nseg = 64 if y.size >= 64 * 32 else max(4, y.size // 32)
    m = y.size // nseg * nseg
    seg = y[:m].reshape(nseg, -1)
    x = np.linspace(-1, 1, seg.shape[1])
    c1 = seg @ x / (x @ x)
    res = seg - seg.mean(1, keepdims=True) - np.outer(c1, x)
    sd = res.std(1)
    # ignore the outermost segments (filter roll-off lowers the noise there)
    edge = max(1, nseg // 16)
    sd = np.sort(sd[edge:-edge]) if nseg > 4 * edge else np.sort(sd)
    v = float(np.percentile(sd, 25))
    return v if v > 0 else 1e-12


def baseline_mask(y: np.ndarray, noise: float, hz_per_pt: float = 0.25) -> np.ndarray:
    """True where the point belongs to baseline (Golotvin-Williams style):
    local standard deviation in a short window must look like pure noise."""
    n = y.size
    w = int(np.clip(4.0 / hz_per_pt, 8, n // 64))            # ~4 Hz window
    m1 = ndimage.uniform_filter1d(y, w)
    m2 = ndimage.uniform_filter1d(y * y, w)
    lsd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    sig = lsd > 2.5 * noise
    # wings of very tall lines: also reject steep local slope
    sl = np.abs(ndimage.uniform_filter1d(np.gradient(m1), w)) * w
    sig |= sl > 3 * noise
    sig = ndimage.binary_dilation(sig, iterations=w)
    return ~sig


def _bin_baseline(y, mask, nbins):
    n = y.size
    edges = np.linspace(0, n, nbins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mm = mask[a:b]
        if mm.sum() >= max(5, 0.15 * (b - a)):
            idx = np.arange(a, b)[mm]
            xs.append(idx.mean())
            ys.append(np.median(y[a:b][mm]))
    return np.asarray(xs), np.asarray(ys)


def baseline_correct(y: np.ndarray, order: int = 4, noise: float | None = None,
                     method: str = "spline", hz_per_pt: float = 0.25):
    """Return (corrected, baseline).  method: 'spline' (default) or 'poly'."""
    if noise is None:
        noise = estimate_noise(y)
    mask = baseline_mask(y, noise, hz_per_pt)
    if mask.sum() < 100:
        return y.copy(), np.zeros_like(y)
    if method == "poly":
        x = np.linspace(-1, 1, y.size)
        c = np.polynomial.chebyshev.chebfit(x[mask], y[mask], order)
        bl = np.polynomial.chebyshev.chebval(x, c)
        return y - bl, bl
    from scipy.interpolate import make_smoothing_spline, PchipInterpolator
    xs, ys = _bin_baseline(y, mask, 256)
    if xs.size < 6:
        return baseline_correct(y, order, noise, "poly", hz_per_pt)
    # robust: drop bins deviating from a coarse smooth curve
    keep = np.ones(xs.size, bool)
    for _ in range(2):
        spl = make_smoothing_spline(xs[keep], ys[keep], lam=(y.size / 30.0) ** 3)
        r = ys - spl(xs)
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-30
        keep = np.abs(r) < 4 * s + 2 * noise
        if keep.sum() < 6:
            keep[:] = True
            break
    spl = make_smoothing_spline(xs[keep], ys[keep], lam=(y.size / 30.0) ** 3)
    xi = np.arange(y.size)
    bl = spl(xi)
    # constant extrapolation outside the covered range
    lo, hi = xs[keep][0], xs[keep][-1]
    bl[xi < lo] = spl(lo)
    bl[xi > hi] = spl(hi)
    return y - bl, bl


def baseline_anchors(ppm, y, points, regions, hz_per_pt):
    """(x, y) anchor values: each point = local median (+-2 Hz); each region is
    sampled every ~10 Hz with local medians."""
    n = y.size
    asc = ppm[0] > ppm[-1]
    w = max(1, int(round(2.0 / hz_per_pt)))

    def idx(p):
        i = int(round((ppm[0] - p) / (ppm[0] - ppm[-1]) * (n - 1))) if asc else \
            int(round((p - ppm[0]) / (ppm[-1] - ppm[0]) * (n - 1)))
        return int(np.clip(i, 0, n - 1))

    xs, ys = [], []
    for p in points:
        i = idx(p)
        a, b = max(0, i - w), min(n, i + w + 1)
        xs.append(float(ppm[i]))
        ys.append(float(np.median(y[a:b])))
    step = max(2, int(round(10.0 / hz_per_pt)))
    for hi, lo in regions:
        i0, i1 = sorted((idx(hi), idx(lo)))
        for a in range(i0, i1 + 1, step):
            b = min(i1 + 1, a + step)
            if b - a < 1:
                continue
            xs.append(float(np.mean(ppm[a:b])))
            ys.append(float(np.median(y[a:b])))
    if not xs:
        return np.array([]), np.array([])
    xs, ys = np.asarray(xs), np.asarray(ys)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    keep = np.r_[True, np.diff(xs) > 1e-12]
    return xs[keep], ys[keep]


def manual_baseline(ppm, y, points, regions, method="spline", order=3, hz_per_pt=0.25):
    """Baseline through user-chosen anchor points / regions.

    spline: PCHIP (shape preserving, no overshoot between anchors);
    poly: least-squares polynomial of the given order.  Constant outside the
    anchored range."""
    xs, ys = baseline_anchors(ppm, y, points, regions, hz_per_pt)
    if xs.size == 0:
        return np.zeros_like(y)
    if xs.size == 1:
        return np.full_like(y, ys[0])
    lo, hi = xs[0], xs[-1]
    xc = np.clip(ppm, lo, hi)
    if method == "poly":
        k = int(min(order, xs.size - 1))
        t = lambda v: 2 * (v - lo) / (hi - lo) - 1
        c = np.polynomial.chebyshev.chebfit(t(xs), ys, k)
        return np.polynomial.chebyshev.chebval(t(xc), c)
    if xs.size == 2:
        return np.interp(xc, xs, ys)
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(xs, ys)(xc)


def auto_baseline_points(ppm, y, hz_per_pt, n=24):
    """Suggest ~n anchor positions spread over signal-free parts."""
    noise = estimate_noise(y)
    mask = baseline_mask(y, noise, hz_per_pt)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    # one anchor per equal-width slice of the spectrum that has baseline points
    edges = np.linspace(0, y.size, n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = idx[(idx >= a) & (idx < b)]
        if sel.size > 0.2 * (b - a):
            out.append(float(ppm[sel[sel.size // 2]]))
    return out


# ---------------------------------------------------------- main routine
def default_lb(nuc: str) -> float:
    return {"1H": 0.3, "19F": 0.5, "13C": 1.0, "31P": 1.0}.get(nuc, 1.0)


def process(spec: Spectrum1D, prm: ProcParams | None = None) -> Processed:
    prm = prm or ProcParams()
    log = []
    if spec.is_fid:
        fid = np.asarray(spec.data, complex)
        lb = default_lb(spec.nucleus) if prm.lb < 0 else prm.lb
        n = (1 << int(np.ceil(np.log2(fid.size)))) * max(1, int(prm.zf))
        buf = np.zeros(n, complex)
        buf[:fid.size] = fid
        gd = max(0.0, spec.group_delay)
        k = int(np.floor(gd))
        frac = gd - k
        # Digital filter: integer part = circular shift (the acausal filter
        # response wraps to negative time = end of buffer), fractional part =
        # linear phase after FT.  No first-point scaling in that case.
        if k:
            buf = np.roll(buf, -k)
        t = np.arange(n, dtype=float)
        if k:
            t[n - k:] -= n
        if lb:
            buf *= np.exp(-np.pi * lb * np.abs(t) / spec.sw)
        if gd == 0:
            buf[0] *= 0.5
        else:
            log.append(f"digital filter: group delay {gd:.3f} pts")
        sp = np.fft.fftshift(np.fft.fft(buf))
        if frac:
            f = np.arange(n) - n // 2
            sp *= np.exp(2j * np.pi * frac * f / n)
        # nmr convention: high ppm on the left (index 0)
        sp = sp[::-1]
        if spec.reverse:
            sp = sp[::-1]
        log.append(f"LB {lb} Hz, SI {n}")
    else:
        sp = np.asarray(spec.data).astype(complex)
        n = sp.size

    size = sp.size
    hz = spec.sw / 2 - np.arange(size) * spec.sw / size
    ppm = spec.center_ppm + hz / spec.obs
    if spec.sw_valid and spec.sw_valid < spec.sw:
        keep = np.abs(hz) <= spec.sw_valid / 2
        sp, ppm = sp[keep], ppm[keep]
        size = sp.size

    # phasing ---------------------------------------------------------
    if prm.pivot_ppm is None:
        piv = float(np.argmax(np.abs(sp))) / size
        prm.pivot_ppm = float(ppm[int(piv * size)])
    piv = float((ppm[0] - prm.pivot_ppm) / (ppm[0] - ppm[-1]))
    if spec.is_fid:
        if prm.auto_phase:
            prm.p0, prm.p1 = auto_phase(sp, piv)
            prm.auto_phase = False  # keep result; user can re-trigger
            log.append(f"auto phase: p0={prm.p0:.1f} p1={prm.p1:.1f}")
        sp = phase(sp, prm.p0, prm.p1, piv)
    real = np.real(sp).copy()
    noise = estimate_noise(real)

    raw_real = real.copy()
    hzpt = abs(ppm[0] - ppm[1]) * spec.obs
    bl = np.zeros_like(real)
    if prm.bl_points or prm.bl_regions:
        bl = manual_baseline(ppm, raw_real, prm.bl_points, prm.bl_regions,
                             prm.bl_manual_method, prm.bl_manual_order, hzpt)
        real = raw_real - bl
        log.append(f"manual baseline ({len(prm.bl_points)} pts, {len(prm.bl_regions)} regions, "
                   f"{prm.bl_manual_method})")
    elif prm.baseline:
        real, bl = baseline_correct(real, prm.baseline_order, noise, prm.baseline_method, hzpt)
        log.append(f"baseline {prm.baseline_method}")

    out = Processed(ppm=ppm, real=real, cplx=sp, params=prm, spec=spec, noise=noise, log=log,
                    raw_real=raw_real, baseline=bl)

    # referencing -------------------------------------------------------
    if prm.ref_mode == "auto":
        shift, note = auto_reference(out, prm.solvent or spec.solvent)
        out.ppm = out.ppm + shift
        out.ref_note = note
        out.ref_applied = float(shift)
    elif prm.ref_mode == "manual":
        out.ppm = out.ppm + prm.ref_shift
        out.ref_applied = float(prm.ref_shift)
        out.ref_note = f"manual {prm.ref_shift:+.4f} ppm"
    return out


def auto_reference(pr: Processed, solvent: str):
    """Snap the residual solvent (or TMS) signal to its literature value."""
    nuc = pr.spec.nucleus
    col = {"1H": 0, "13C": 1}.get(nuc)
    sname = norm_solvent(solvent)
    if col is None or not sname:
        return 0.0, "no reference (solvent unknown)"
    target = SOLVENTS[sname][col]
    if target is None:
        return 0.0, f"{sname}: no {nuc} reference"
    win = 0.25 if nuc == "1H" else 2.5
    sel = np.where(np.abs(pr.ppm - target) < win)[0]
    if sel.size < 5:
        return 0.0, "reference window outside spectrum"
    y = pr.real[sel]
    from scipy.signal import find_peaks
    pk, _ = find_peaks(y, height=max(0.05 * y.max(), 10 * pr.noise))
    if pk.size == 0:
        return 0.0, f"{sname} signal not found"
    # take the significant line nearest the expected position; for solvent
    # multiplets (DMSO/CD3OD quintet, CDCl3 13C triplet) use the centre of the
    # lines belonging to the same group (within the group's span).
    hs = y[pk]
    big = pk[hs > 0.15 * hs.max()]
    ppm_sel = pr.ppm[sel]
    near = big[np.argmin(np.abs(ppm_sel[big] - target))]
    step_hz = 3.0 if nuc == "1H" else 40.0  # max line spacing inside a solvent multiplet
    lim = step_hz / pr.spec.obs
    cand = pk[y[pk] > 0.08 * y[near]]
    cp = ppm_sel[cand]
    grp = {int(near)}
    grow = True
    while grow:
        grow = False
        cur = ppm_sel[sorted(grp)]
        for c, v in zip(cand, cp):
            if int(c) not in grp and np.min(np.abs(cur - v)) <= lim:
                grp.add(int(c))
                grow = True
    grp = np.array(sorted(grp, key=lambda k: -ppm_sel[k]))
    if grp.size >= 3 and grp.size % 2 == 1:
        near = grp[grp.size // 2]
    i = sel[near]
    # parabolic interpolation
    if 0 < i < len(pr.real) - 1:
        a, b, c = pr.real[i - 1:i + 2]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
    else:
        off = 0.0
    step = pr.ppm[1] - pr.ppm[0]
    found = pr.ppm[i] + off * step
    shift = target - found
    return float(shift), f"{sname} {nuc} → {target:.2f} ppm (shift {shift:+.3f})"
