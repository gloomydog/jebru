"""Peak picking, integration, first-order multiplet analysis and report text."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from .processing import Processed, estimate_noise, norm_solvent, SOLVENTS
from . import impurities as IMP

# residual water (1H) per solvent (Fulmer 2010)
WATER = {"CDCl3": 1.56, "DMSO-d6": 3.33, "CD3OD": 4.87, "D2O": 4.79, "Acetone-d6": 2.84,
         "C6D6": 0.40, "CD3CN": 2.13, "CD2Cl2": 1.52, "THF-d8": 2.46, "Toluene-d8": 0.43,
         "C6D5Cl": 1.03, "TFE-d3": 3.66, "Pyridine-d5": 4.96}

MULT_NAME = {1: "d", 2: "t", 3: "q", 4: "p", 5: "h", 6: "hept"}


@dataclass
class Peak:
    ppm: float
    height: float
    index: int
    note: str = ""          # impurity / solvent label


@dataclass
class Region:
    start: float            # high ppm
    stop: float             # low ppm
    value: float = 0.0      # raw integral
    nH: float = 0.0         # normalized
    mult: str = "m"
    J: list = field(default_factory=list)
    center: float = 0.0
    lines: list = field(default_factory=list)   # Peak list
    tag: str = ""           # "", "solvent", "H2O"
    include: bool = True

    def label(self):
        return self.mult


# ------------------------------------------------------------- utilities
def spectrum_noise(pr: Processed) -> float:
    return estimate_noise(pr.real)


def _interp_peak(y, i):
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den != 0:
            off = 0.5 * (a - c) / den
            return float(np.clip(off, -0.5, 0.5)), float(b - 0.25 * (a - c) * off)
    return 0.0, float(y[i])


def pick_peaks(pr: Processed, threshold: float | None = None, lo: float | None = None,
               hi: float | None = None, snr: float = 5.0) -> list[Peak]:
    """Return peaks (descending ppm).  threshold: absolute height; default snr*noise."""
    y = pr.real
    noise = spectrum_noise(pr)
    thr = threshold if threshold is not None else snr * noise
    i0, i1 = 0, len(y)
    if hi is not None:
        i0 = int(max(0, np.floor(pr.ppm_to_index(hi))))
    if lo is not None:
        i1 = int(min(len(y), np.ceil(pr.ppm_to_index(lo)) + 1))
    seg = y[i0:i1]
    if seg.size < 3:
        return []
    min_dist = max(1, int(0.15 / pr.hz_per_pt))   # lines closer than 0.15 Hz are unresolvable
    idx, _ = find_peaks(seg, height=thr, distance=min_dist, prominence=thr * 0.5)
    step = pr.ppm[1] - pr.ppm[0]
    out = []
    for k in idx:
        i = i0 + k
        off, h = _interp_peak(y, i)
        out.append(Peak(ppm=float(pr.ppm[i] + off * step), height=h, index=i))
    return out


def integrate(pr: Processed, start: float, stop: float) -> float:
    a, b = sorted([pr.ppm_to_index(start), pr.ppm_to_index(stop)])
    a, b = int(max(0, np.floor(a))), int(min(len(pr.real) - 1, np.ceil(b)))
    if b <= a:
        return 0.0
    return float(np.sum(pr.real[a:b + 1]) * abs(pr.ppm[1] - pr.ppm[0]))


# --------------------------------------------------------- auto integrate
def auto_regions(pr: Processed, snr: float = 6.0, min_frac: float = 0.005,
                 rel: float = 0.003) -> list[Region]:
    """Automatic integral regions.  A point is 'signal' if the smoothed
    spectrum exceeds snr*noise AND rel*max (so Lorentzian wings of very
    strong lines do not glue neighbouring multiplets together)."""
    y = pr.real
    hzpt = pr.hz_per_pt
    sig = max(1.0, 0.8 / hzpt)            # ~0.8 Hz gaussian smoothing
    sm = ndimage.gaussian_filter1d(y, sig)
    sm_noise = estimate_noise(sm)
    mask = sm > max(snr * sm_noise, rel * sm.max())
    # extend to cover Lorentzian wings and merge close groups
    ext = int(max(3, 4.0 / hzpt))         # +-4 Hz
    mask = ndimage.binary_dilation(mask, iterations=ext)
    gap = int(max(2, 3.0 / hzpt))
    mask = ndimage.binary_closing(mask, iterations=gap)
    lab, n = ndimage.label(mask)
    regs = []
    for k in range(1, n + 1):
        ii = np.where(lab == k)[0]
        r = Region(start=float(pr.ppm[ii[0]]), stop=float(pr.ppm[ii[-1]]))
        r.value = integrate(pr, r.start, r.stop)
        regs.append(r)
    if not regs:
        return []
    vmax = max(r.value for r in regs)
    regs = [r for r in regs if r.value > min_frac * vmax]
    return regs


def _split_at(pr: Processed, r: Region, ppm_a: float, ppm_b: float, depth: float):
    """Split region r at the spectrum minimum between ppm_a and ppm_b if the
    minimum is below depth*(smaller of the two line heights)."""
    i, j = sorted([int(round(pr.ppm_to_index(ppm_a))), int(round(pr.ppm_to_index(ppm_b)))])
    if j - i < 3:
        return None
    seg = pr.real[i:j + 1]
    k = i + int(np.argmin(seg))
    lim = depth * min(pr.real[i], pr.real[j])
    if pr.real[k] > lim:
        return None
    cut = float(pr.ppm[k])
    r1 = Region(start=r.start, stop=cut)
    r2 = Region(start=cut, stop=r.stop)
    for x in (r1, r2):
        x.value = integrate(pr, x.start, x.stop)
    return r1, r2


def split_regions(pr: Processed, regs: list[Region], gap_hz: float = 21.0) -> list[Region]:
    """Split regions whose lines form separate clusters (gap > gap_hz)."""
    out = []
    stack = list(regs)
    while stack:
        r = stack.pop(0)
        pk = pick_peaks(pr, lo=r.stop, hi=r.start)
        if len(pk) >= 2:
            hm = max(p.height for p in pk)
            pk = [p for p in pk if p.height > 0.03 * hm]
            pk.sort(key=lambda p: -p.ppm)
            gaps = [(pk[i].ppm - pk[i + 1].ppm) * pr.spec.obs for i in range(len(pk) - 1)]
            done = False
            for gi in np.argsort(gaps)[::-1]:
                if gaps[gi] < gap_hz:
                    break
                sp = _split_at(pr, r, pk[gi].ppm, pk[gi + 1].ppm, 0.25)
                if sp:
                    stack = list(sp) + stack
                    done = True
                    break
            if done:
                continue
        out.append(r)
    return out


def split_solvent(pr: Processed, regs: list[Region], solvent: str | None = None) -> list[Region]:
    """If the residual solvent line sits inside a region with other lines,
    cut it out into its own region."""
    s = norm_solvent(solvent or pr.params.solvent or pr.spec.solvent)
    if pr.spec.nucleus != "1H" or not s or SOLVENTS[s][0] is None:
        return regs
    sv = SOLVENTS[s][0]
    out = []
    for r in regs:
        if not (r.stop < sv < r.start) or r.start - r.stop < 0.03:
            out.append(r)
            continue
        pk = sorted(pick_peaks(pr, lo=r.stop, hi=r.start), key=lambda p: -p.ppm)
        si = [k for k, p in enumerate(pk) if abs(p.ppm - sv) < 0.012]
        if not si or len(pk) < 2:
            out.append(r)
            continue
        k = si[0]
        pieces = [r]
        # cut on the high-ppm side, then on the low-ppm side
        if k > 0:
            sp = _split_at(pr, pieces[-1], pk[k - 1].ppm, pk[k].ppm, 0.5)
            if sp:
                pieces = [sp[0], sp[1]]
        if k < len(pk) - 1:
            sp = _split_at(pr, pieces[-1], pk[k].ppm, pk[k + 1].ppm, 0.5)
            if sp:
                pieces = pieces[:-1] + [sp[0], sp[1]]
        out.extend(pieces)
    return out


def tag_regions(pr: Processed, regs: list[Region], solvent: str | None = None):
    s = norm_solvent(solvent or pr.params.solvent or pr.spec.solvent)
    if pr.spec.nucleus != "1H" or not s:
        return
    sv = SOLVENTS[s][0]
    w = WATER.get(s)
    for r in regs:
        width = r.start - r.stop
        pk = pick_peaks(pr, lo=r.stop, hi=r.start) if sv is not None else []
        top = max(pk, key=lambda p: p.height) if pk else None
        if top is not None and r.stop - 0.01 <= sv <= r.start + 0.01 and width < 0.2 \
                and abs(top.ppm - sv) < 0.006 \
                and sum(p.height > 0.1 * top.height and abs(p.ppm - sv) > 0.012 for p in pk) == 0:
            r.tag, r.include = "solvent", False
        elif w is not None and r.stop - 0.06 <= w <= r.start + 0.06 and width < 1.0:
            pk = pick_peaks(pr, lo=r.stop, hi=r.start)
            if pk:
                top = max(pk, key=lambda p: p.height)
                n_big = sum(p.height > 0.2 * top.height for p in pk)
                if abs(top.ppm - w) < 0.08 and (n_big == 1 or _fwhm_hz(pr, top) > 3):
                    r.tag, r.include = "H2O", False
        if s == "CDCl3" and r.stop - 0.01 <= 0.0 <= r.start + 0.01 and width < 0.1:
            r.tag, r.include = "TMS", False


def normalize(regs: list[Region], ref: Region | None = None, ref_value: float | None = None):
    """Scale integrals to proton counts.

    Without an explicit reference the scale is chosen so that the large
    regions become (close to) integers with the fewest total protons.
    """
    use = [r for r in regs if r.include and r.value > 0]
    if not use:
        return
    if ref is not None and ref_value:
        scale = ref.value / ref_value
    else:
        vals = np.array([r.value for r in use])
        total = vals.sum()
        cands = sorted({v / m for v in vals for m in (1, 2, 3, 4, 6, 9)}, reverse=True)
        scored = []
        for sc in cands:
            n = vals / sc
            big = n >= 0.5
            if vals[big].sum() < 0.9 * total:
                continue
            scored.append((sc, float(np.mean(np.abs(n[big] - np.round(n[big]))))))
        best = None
        if scored:
            emin = min(e for _, e in scored)
            # fewest protons whose integer misfit is close to the best one
            best = next(sc for sc, e in scored if e <= emin * 1.5 + 0.03)
        scale = best or vals.min()
    for r in regs:
        r.nH = r.value / scale


# ------------------------------------------------------ multiplet analysis
def _try_split(pos, amp, J, tol, tol_int):
    r = amp.astype(float).copy()
    eps = 0.08 * amp.max()
    sub_p, sub_a = [], []
    for i in range(len(pos)):
        if r[i] <= eps:
            continue
        a = r[i]
        cand = np.where(np.abs(pos - (pos[i] + J)) < tol)[0]
        cand = cand[cand > i]
        if cand.size == 0:
            return None
        j = cand[np.argmax(r[cand])]
        if r[j] < a * (1 - tol_int):
            return None
        r[j] = max(0.0, r[j] - a)
        r[i] = 0.0
        sub_p.append(pos[i])
        sub_a.append(a)
    if np.any(r > eps * 1.5):
        return None
    return np.asarray(sub_p), np.asarray(sub_a)


def _jtree(pos, amp, tol, tol_int, jmax, depth=0):
    if len(pos) == 1:
        return []
    if depth > 5:
        return None
    for k in range(1, len(pos)):
        J = pos[k] - pos[0]
        if J < 0.6:
            continue
        if J > jmax:
            break
        s = _try_split(pos, amp, J, tol, tol_int)
        if s is None:
            continue
        rest = _jtree(s[0], s[1], tol, tol_int, jmax, depth + 1)
        if rest is not None:
            return [J] + rest
    return None


def analyze_multiplet(pr: Processed, reg: Region, snr: float = 5.0):
    """Fill reg.mult / reg.J / reg.center / reg.lines (1H first-order analysis)."""
    noise = spectrum_noise(pr)
    peaks = pick_peaks(pr, lo=reg.stop, hi=reg.start, snr=snr)
    if peaks:
        hmax = max(p.height for p in peaks)
        peaks = [p for p in peaks if p.height > 0.04 * hmax]
    reg.lines = peaks
    if not peaks:
        reg.mult, reg.J, reg.center = "m", [], (reg.start + reg.stop) / 2
        return reg
    main = [p for p in peaks if p.height >= 0.10 * max(q.height for q in peaks)]
    pos = np.array([p.ppm for p in main]) * pr.spec.obs     # Hz
    amp = np.array([p.height for p in main])
    order = np.argsort(pos)
    pos, amp = pos[order], amp[order]
    reg.center = float(np.sum(pos * amp) / np.sum(amp) / pr.spec.obs)
    if len(main) == 1:
        fwhm = _fwhm_hz(pr, main[0])
        reg.mult = "br s" if fwhm > 4.0 else "s"
        reg.J = []
        reg.center = main[0].ppm
        return reg
    if len(main) > 20:
        reg.mult, reg.J = "m", []
        return reg
    tol = max(0.35, 2.5 * pr.hz_per_pt)
    js = _jtree(pos, amp, tol, 0.45, jmax=20.0)
    if js is None:
        reg.mult, reg.J = "m", []
        return reg
    js = sorted(js, reverse=True)
    groups = []
    for j in js:
        if groups and abs(groups[-1][0] / groups[-1][1] - j) < max(0.6, 0.08 * j):
            groups[-1][0] += j
            groups[-1][1] += 1
        else:
            groups.append([j, 1])
    if any(g[1] > 6 for g in groups):
        reg.mult, reg.J = "m", []
        return reg
    reg.mult = "".join(MULT_NAME[g[1]] for g in groups)
    reg.J = [round(g[0] / g[1], 1) for g in groups]
    # symmetric multiplet: centre = midpoint of outer lines
    reg.center = float((pos[0] + pos[-1]) / 2 / pr.spec.obs)
    return reg


def _fwhm_hz(pr: Processed, pk: Peak) -> float:
    y = pr.real
    i = pk.index
    half = y[i] / 2
    l = i
    while l > 0 and y[l] > half:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half:
        r += 1
    return (r - l) * pr.hz_per_pt


# ------------------------------------------------------------ impurities
def _solvent_of(pr: Processed) -> str:
    return norm_solvent(pr.params.solvent or pr.spec.solvent)


def tag_impurities_1h(pr: Processed, regs: list[Region]):
    """Label regions that match common laboratory impurities for the solvent.

    Confirmed multi-signal compounds (all signals, right multiplicities and
    integral ratio, e.g. EtOAc 4.12 q / 2.05 s / 1.26 t in CDCl3) are excluded
    from the report.  Single-signal matches get a '?' label; they are excluded
    only if small (see exclude_small_possible)."""
    cand = [r for r in regs if not r.tag and r.center]
    matches = IMP.find_impurities([(r.center, r.mult, r.value) for r in cand], _solvent_of(pr), "1H")
    for m in matches:
        for h in m.hits:
            for r in cand:
                if r.center == h.found and not r.tag:
                    r.tag = m.label
                    if m.status == "confirmed" and (r.start - r.stop) < 0.25:
                        r.include = False
    return matches


def exclude_small_possible(regs: list[Region], limit: float = 0.5):
    """'?' matches that are small are treated as impurities; big ones lose the
    label (they are most likely real signals of the compound)."""
    changed = False
    for r in regs:
        if r.tag.endswith("?"):
            if r.nH < limit and r.include:
                r.include = False
                changed = True
            elif r.nH >= limit:
                r.tag = ""
    return changed


def build_regions(pr: Processed, snr: float = 6.0):
    """Automatic 1H regions with multiplet analysis, solvent/water/impurity
    tagging and normalisation.  Returns (regions, impurity_matches)."""
    regs = auto_regions(pr, snr=snr)
    regs = split_regions(pr, regs)
    regs = split_solvent(pr, regs)
    tag_regions(pr, regs)
    for r in regs:
        analyze_multiplet(pr, r)
    matches = tag_impurities_1h(pr, regs)
    normalize(regs)
    if exclude_small_possible(regs):
        normalize(regs)
    # keep only matches that are still attached to a region
    labels = {r.tag for r in regs}
    matches = [m for m in matches if m.label in labels]
    return regs, matches


def tag_impurities_13c(pr: Processed, peaks: list[Peak]):
    matches = IMP.find_impurities([p.ppm for p in peaks], _solvent_of(pr), "13C")
    for m in matches:
        for h in m.hits:
            for p in peaks:
                if h.found is not None and p.ppm == h.found and not p.note:
                    p.note = m.label
    return matches


def impurity_summary(matches) -> str:
    if not matches:
        return ""
    parts = []
    for m in matches:
        ppm = ", ".join(f"{h.found:.2f}" for h in m.hits if h.found is not None)
        parts.append(f"{IMP.short(m.name)} ({'confirmed' if m.status == 'confirmed' else 'possible'}: {ppm})")
    return "Detected solvents/impurities: " + "; ".join(parts)


# ------------------------------------------------------------- reporting
def _solvent_label(pr: Processed):
    s = norm_solvent(pr.params.solvent or pr.spec.solvent)
    return s or (pr.spec.solvent or "?")


def nominal_freq(pr: Processed) -> int:
    return int(round(pr.spec.obs))


def report_1h(pr: Processed, regs: list[Region], digits: int = 2) -> str:
    items = []
    for r in sorted([r for r in regs if r.include], key=lambda r: -(r.center or r.start)):
        n = int(round(r.nH)) if r.nH else 0
        if n == 0:
            continue
        if r.mult == "m" or not r.lines:
            lines = [p.ppm for p in r.lines] or [r.start, r.stop]
            hi, lo = max(lines), min(lines)
            if hi - lo < 10 ** -digits:
                d = f"{hi:.{digits}f}"
            else:
                d = f"{hi:.{digits}f}–{lo:.{digits}f}"
            items.append(f"{d} (m, {n}H)")
        else:
            jtxt = ""
            if r.J:
                jtxt = ", J = " + ", ".join(f"{j:.1f}" for j in r.J) + " Hz"
            items.append(f"{r.center:.{digits}f} ({r.mult}{jtxt}, {n}H)")
    return f"¹H NMR ({nominal_freq(pr)} MHz, {_solvent_label(pr)}) δ " + ", ".join(items) + "."


def carbon_peaks(pr: Processed, snr: float = 5.0, threshold: float | None = None) -> list[Peak]:
    pk = pick_peaks(pr, threshold=threshold, snr=snr)
    s = norm_solvent(pr.params.solvent or pr.spec.solvent)
    if s and SOLVENTS[s][1] is not None:
        c = SOLVENTS[s][1]
        pk = [p for p in pk if abs(p.ppm - c) > 1.2]
    return pk


def report_13c(pr: Processed, peaks: list[Peak], digits: int = 1) -> str:
    # confirmed impurities (label without '?') are left out of the report
    vals = sorted({round(p.ppm, digits) for p in peaks if not p.note or p.note.endswith("?")}, reverse=True)
    return f"¹³C NMR ({nominal_freq(pr)} MHz, {_solvent_label(pr)}) δ " + \
        ", ".join(f"{v:.{digits}f}" for v in vals) + "."


def full_analysis(pr: Processed, snr: float = 6.0):
    """Convenience: returns (regions, peaks, report_text, impurity_matches)."""
    if pr.spec.nucleus == "1H":
        regs, matches = build_regions(pr, snr=snr)
        peaks = [p for r in regs for p in r.lines]
        return regs, peaks, report_1h(pr, regs), matches
    peaks = carbon_peaks(pr, snr=snr)
    if pr.spec.nucleus == "13C":
        matches = tag_impurities_13c(pr, peaks)
        return [], peaks, report_13c(pr, peaks), matches
    vals = ", ".join(f"{p.ppm:.2f}" for p in sorted(peaks, key=lambda p: -p.ppm))
    return [], peaks, f"{pr.spec.nucleus} NMR ({nominal_freq(pr)} MHz, {_solvent_label(pr)}) δ {vals}.", []
