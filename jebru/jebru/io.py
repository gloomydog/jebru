"""Data loading for JEOL Delta (.jdf) and Bruker TopSpin (directory / .zip / fid)."""
from __future__ import annotations

import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field

import numpy as np
import nmrglue as ng

NUC_LABEL = {
    "1H": "1H", "PROTON": "1H", "H1": "1H",
    "13C": "13C", "CARBON13": "13C", "C13": "13C", "CARBON": "13C",
    "19F": "19F", "FLUORINE19": "19F", "F19": "19F",
    "31P": "31P", "PHOSPHORUS31": "31P", "P31": "31P",
    "15N": "15N", "NITROGEN15": "15N", "N15": "15N",
    "2H": "2H", "DEUTERIUM": "2H", "11B": "11B", "BORON11": "11B",
    "29SI": "29Si", "SILICON29": "29Si",
}


def norm_nucleus(s: str) -> str:
    s = str(s or "").strip()
    return NUC_LABEL.get(s.upper().replace("-", ""), s)


@dataclass
class Spectrum1D:
    """Raw 1D dataset in a vendor-neutral form."""
    data: np.ndarray            # complex FID (time domain) or real/complex spectrum
    sw: float                   # spectral width [Hz]
    obs: float                  # observe frequency [MHz]
    center_ppm: float           # ppm at the centre of the spectral window
    nucleus: str = "1H"
    solvent: str = ""
    vendor: str = ""
    path: str = ""
    title: str = ""
    group_delay: float = 0.0    # digital filter group delay [points]
    is_fid: bool = True         # False if the file already holds a spectrum
    reverse: bool = False       # spectrum must be reversed after FT
    sw_valid: float = 0.0       # usable bandwidth [Hz] (filter pass band), 0 = all
    params: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        if self.vendor == "Bruker":
            p = self.path.split("::")[-1].rstrip("/")
            return f"{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)}".strip("_")
        return os.path.basename(self.path.rstrip("/")) or self.title


class LoadError(Exception):
    pass


# --------------------------------------------------------------------- JEOL
def _jeol_group_delay(p: dict) -> float:
    """Group delay of the JEOL Delta digital filter, from 'orders'/'factors'.

    JEOL stores e.g. orders='2 54 73', factors='8 2' (nmrglue may hand them
    over with the blanks removed: 25473 / 82).  First digit of 'orders' is the
    number of filter stages.  gd = 1/2 * sum_i (order_i - 1) / prod(factors_i..)
    """
    try:
        o = str(p["orders"]).strip()
        f = str(p["factors"]).strip()
        if " " in o:
            ol = [int(v) for v in o.split()]
            n, orders = ol[0], ol[1:]
        else:
            n = int(o[0])
            rest = o[1:]
            k = len(rest) // n
            orders = [int(rest[i * k:(i + 1) * k]) for i in range(n)]
        factors = [int(v) for v in f.split()] if " " in f else [int(c) for c in f]
        if len(orders) != n or len(factors) != n:
            raise ValueError
        s = 0.0
        for i in range(n):
            s += (orders[i] - 1) / np.prod(factors[i:])
        return s / 2.0
    except Exception:
        return float("nan")


def _estimate_group_delay(fid: np.ndarray) -> float:
    """Fallback: position where the FID 'switches on'."""
    a = np.abs(fid[:256])
    if a.max() == 0:
        return 0.0
    return float(np.argmax(a > 0.3 * a.max()))


def load_jeol(path: str) -> Spectrum1D:
    dic, data = ng.jeol.read(path)
    if data.ndim != 1:
        raise LoadError(f"{data.ndim}D data (only 1D is supported)")
    p = dic["parameters"]
    h = dic["header"]
    nuc = norm_nucleus(p.get("x_domain", ""))
    units = h.get("units", [("none", 1, "second")])[0]
    is_fid = "second" in str(units)
    obs = float(p.get("x_freq", 0)) / 1e6
    if is_fid:
        sw = float(p["x_sweep"])
        center = float(p.get("x_offset", 0.0))
        gd = _jeol_group_delay(p) if str(p.get("digital_filter", "TRUE")).upper() == "TRUE" else 0.0
        if not np.isfinite(gd):
            gd = _estimate_group_delay(data)
    else:
        # already processed spectrum: axis given in ppm
        start, stop = float(h["data_axis_start"][0]), float(h["data_axis_stop"][0])
        sw = abs(start - stop) * obs
        center = (start + stop) / 2
        gd = 0.0
        data = np.real(data).astype(float)
        if start < stop:
            data = data[::-1]
    return Spectrum1D(
        data=np.asarray(data), sw=sw, obs=obs, center_ppm=center, nucleus=nuc,
        solvent=str(p.get("solvent", "")), vendor="JEOL", path=path,
        title=str(h.get("title", "")), group_delay=gd, is_fid=is_fid,
        sw_valid=float(p.get("x_sweep_clipped", 0) or 0) if is_fid else 0.0,
        params={k: v for k, v in p.items() if np.isscalar(v) or isinstance(v, str)},
    )


# ------------------------------------------------------------------- Bruker
def _find_bruker_expdir(path: str) -> str:
    """Accept experiment dir, fid file, pdata/N dir, or a parent dir."""
    if os.path.isfile(path):
        path = os.path.dirname(path)
    p = path.rstrip("/")
    for cand in (p, os.path.dirname(p), os.path.dirname(os.path.dirname(p))):
        if os.path.isfile(os.path.join(cand, "acqus")):
            return cand
    # search below (e.g. unzipped archive)
    hits = []
    for root, dirs, files in os.walk(path):
        if "acqus" in files and ("fid" in files or "ser" in files):
            hits.append(root)
    if not hits:
        raise LoadError("Bruker acqus file not found")
    hits.sort()
    return hits[0]


def list_bruker_experiments(path: str) -> list[str]:
    hits = []
    for root, dirs, files in os.walk(path):
        if "acqus" in files and ("fid" in files or "ser" in files):
            hits.append(root)
    return sorted(hits, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])


_BRUKER_DSP = ng.fileio.bruker.bruker_dsp_table


def _bruker_group_delay(a: dict) -> float:
    gd = a.get("GRPDLY", -1)
    if gd is not None and gd > 0:
        return float(gd)
    dspfvs, decim = int(a.get("DSPFVS", 0)), a.get("DECIM", 1)
    if dspfvs in _BRUKER_DSP:
        tab = _BRUKER_DSP[dspfvs]
        for k in (decim, int(decim) if float(decim).is_integer() else decim):
            if k in tab:
                return float(tab[k])
    return 0.0


def load_bruker(path: str) -> Spectrum1D:
    exp = _find_bruker_expdir(path)
    if os.path.isfile(os.path.join(exp, "ser")) and not os.path.isfile(os.path.join(exp, "fid")):
        raise LoadError("2D (ser) data (only 1D is supported)")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dic, data = ng.bruker.read(exp)
    if data.ndim != 1:
        raise LoadError(f"{data.ndim}D data (only 1D is supported)")
    a = dic["acqus"]
    sfo1, bf1 = float(a["SFO1"]), float(a["BF1"])
    sf = bf1
    procs_dir = os.path.join(exp, "pdata", "1")
    if os.path.isfile(os.path.join(procs_dir, "procs")):
        try:
            sf = float(ng.bruker.read_procs_file(procs_dir)["procs"]["SF"]) or bf1
        except Exception:
            pass
    sw = float(a["SW_h"])
    center = (sfo1 - sf) / sf * 1e6
    title = ""
    tpath = os.path.join(procs_dir, "title")
    if os.path.isfile(tpath):
        with open(tpath, errors="ignore") as fh:
            title = fh.read().strip()
    return Spectrum1D(
        data=np.asarray(data), sw=sw, obs=sf, center_ppm=center,
        nucleus=norm_nucleus(a.get("NUC1", "1H")), solvent=str(a.get("SOLVENT", "")),
        vendor="Bruker", path=exp,
        title=(title.splitlines()[0] if title else "") or
        f"{os.path.basename(os.path.dirname(exp))}/{os.path.basename(exp)}",
        group_delay=_bruker_group_delay(a), is_fid=True,
        params={k: v for k, v in a.items() if np.isscalar(v) or isinstance(v, str)},
    )


# ---------------------------------------------------------------- dispatch
_TMPDIRS: list[tempfile.TemporaryDirectory] = []


def load(path: str) -> Spectrum1D:
    """Load a 1D NMR dataset (JEOL .jdf, Bruker dir / fid / .zip)."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise LoadError(f"not found: {path}")
    low = path.lower()
    if low.endswith(".jdf"):
        return load_jeol(path)
    if low.endswith(".zip"):
        td = tempfile.TemporaryDirectory(prefix="jebru_")
        _TMPDIRS.append(td)
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():  # guard against path traversal
                dest = os.path.realpath(os.path.join(td.name, m))
                if not dest.startswith(os.path.realpath(td.name) + os.sep):
                    raise LoadError("invalid zip file")
            z.extractall(td.name)
        jdfs = [os.path.join(r, f) for r, _, fs in os.walk(td.name) for f in fs if f.lower().endswith(".jdf")]
        if jdfs:
            return load_jeol(sorted(jdfs)[0])
        exps = [e for e in list_bruker_experiments(td.name) if os.path.isfile(os.path.join(e, "fid"))]
        if not exps:
            raise LoadError("no 1D data in the zip file")
        s = load_bruker(exps[0])
        s.path = path + "::" + os.path.relpath(exps[0], td.name)
        return s
    return load_bruker(path)
