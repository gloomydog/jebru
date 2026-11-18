"""Synthetic FID tests: known shifts / couplings / integrals, with a digital-filter delay."""
import numpy as np
import pytest

from jebru.io import Spectrum1D
from jebru import processing as P, analysis as A

SW, OBS, N = 6000.0, 400.0, 32768


def make(lines, gd=20.4, p0=37.0, noise=2e-4, seed=0, solvent=""):
    t = np.arange(N) / SW
    fid = np.zeros(N, complex)
    for ppm, amp in lines:
        f = (ppm - 5.0) * OBS
        fid += amp * np.exp(2j * np.pi * f * t - t * np.pi * 0.8)
    fid *= np.exp(1j * np.deg2rad(p0))
    # emulate digital-filter group delay: time-shift by gd points
    n2 = 2 * N
    sp = np.fft.fft(np.r_[fid, np.zeros(N)])
    k = np.fft.fftfreq(n2) * n2
    fid = np.fft.ifft(sp * np.exp(-2j * np.pi * gd * k / n2))[:N]
    fid += noise * (np.random.default_rng(seed).normal(size=N) + 1j * np.random.default_rng(seed + 1).normal(size=N))
    return Spectrum1D(data=fid, sw=SW, obs=OBS, center_ppm=5.0, nucleus="1H", solvent=solvent, group_delay=gd)


def multiplet(center, js, nH):
    pos = np.array([center * OBS])
    amp = np.array([1.0])
    for j in js:
        pos = np.r_[pos - j / 2, pos + j / 2]
        amp = np.r_[amp, amp] / 2
    return [(p / OBS, a * nH) for p, a in zip(pos, amp)]


@pytest.fixture(scope="module")
def result():
    lines = (multiplet(7.20, [8.0], 2) + multiplet(4.10, [7.0, 7.0, 7.0], 2) +
             multiplet(2.50, [], 3) + multiplet(1.25, [7.0, 7.0], 3) + multiplet(6.05, [15.6, 6.9], 1))
    spec = make(lines)
    pr = P.process(spec, P.ProcParams(ref_mode="none"))
    regs, peaks, rep, _ = A.full_analysis(pr)
    return pr, regs, rep


def find(regs, ppm):
    return min(regs, key=lambda r: abs(r.center - ppm))


def test_phase_and_positions(result):
    pr, regs, rep = result
    assert pr.real.min() > -0.02 * pr.real.max()       # absorptive, no big negative lobes
    for ppm in (7.20, 4.10, 2.50, 1.25, 6.05):
        assert abs(find(regs, ppm).center - ppm) < 0.003


@pytest.mark.parametrize("ppm,mult,js,nh", [
    (7.20, "d", [8.0], 2), (4.10, "q", [7.0], 2), (2.50, "s", [], 3),
    (1.25, "t", [7.0], 3), (6.05, "dd", [15.6, 6.9], 1)])
def test_multiplets(result, ppm, mult, js, nh):
    pr, regs, rep = result
    r = find(regs, ppm)
    assert r.mult == mult
    assert np.allclose(r.J, js, atol=0.2)
    assert round(r.nH) == nh


def test_report(result):
    pr, regs, rep = result
    assert rep.startswith("¹H NMR (400 MHz")
    assert "6.05 (dd, J = 15.6, 6.9 Hz, 1H)" in rep


def test_impurities_cdcl3():
    """Compound + EtOAc (all three signals) + small acetone singlet + CHCl3."""
    J = 7.1
    comp = multiplet(7.50, [8.0], 2) + multiplet(3.85, [], 3) + multiplet(2.80, [7.5, 7.5], 2)
    etoac = multiplet(4.12, [J, J, J], 0.4) + multiplet(2.05, [], 0.6) + multiplet(1.26, [J, J], 0.6)
    other = multiplet(2.17, [], 0.15) + multiplet(7.26, [], 0.3)
    spec = make(comp + etoac + other, solvent="CDCl3")
    pr = P.process(spec, P.ProcParams(ref_mode="auto"))
    regs, peaks, rep, matches = A.full_analysis(pr)
    names = {m.name: m.status for m in matches}
    assert names.get("ethyl acetate") == "confirmed"
    assert names.get("acetone") == "possible"
    for ppm in (4.12, 2.05, 1.26, 2.17):
        r = find(regs, ppm)
        assert r.tag in ("EtOAc", "acetone?") and not r.include
    # compound signals stay in the report with correct proton counts
    assert "7.50 (d, J = 8.0 Hz, 2H)" in rep and "3.85 (s, 3H)" in rep and "2.80 (t, J = 7.5 Hz, 2H)" in rep
    for x in ("4.12", "2.05", "1.26", "2.17", "7.26"):
        assert x not in rep


def test_impurity_table_matches_fulmer_2010():
    """Spot checks against Fulmer et al., Organometallics 2010, Tables 1/2."""
    from jebru import impurities as I
    t = I.table_for("CDCl3", "1H")
    assert [r[1] for r in t["ethyl acetate"]] == [2.05, 4.12, 1.26]
    assert t["dichloromethane"][0][1] == 5.30 and t["n-hexane"][0][1] == 0.88
    assert I.table_for("THF-d8", "1H")["acetone"][0][1] == 2.05
    assert I.table_for("CD2Cl2", "1H")["diethyl ether"][1][1] == 3.43
    assert I.table_for("DMSO-d6", "1H")["silicone grease"][0][1] == -0.06
    assert I.table_for("Toluene-d8", "1H")["H grease"][0][1] == 0.925      # 0.89-0.96
    assert I.table_for("D2O", "13C")["methanol"][0][1] == 49.50
    assert [r[1] for r in I.table_for("TFE-d3", "13C")["acetone"]] == [214.98, 32.35]
    assert "chloroform" not in I.table_for("CDCl3", "1H")                 # own residual signal
