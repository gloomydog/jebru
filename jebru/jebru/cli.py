"""Command line / batch processing.

    jebru-cli sample.jdf                 # report to stdout
    jebru-cli expdir/ --png out.png --csv peaks.csv
    jebru-cli *.jdf --outdir results/    # batch
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

from . import io, processing as P, analysis as A


def run_one(path, args):
    spec = io.load(path)
    prm = P.ProcParams(lb=args.lb, zf=args.zf, baseline=not args.no_baseline,
                       ref_mode="none" if args.no_ref else "auto", solvent=args.solvent or "")
    if args.p0 is not None or args.p1 is not None:
        prm.auto_phase = False
        prm.p0, prm.p1 = args.p0 or 0.0, args.p1 or 0.0
    pr = P.process(spec, prm)
    regs, peaks, report, matches = A.full_analysis(pr, snr=args.snr)
    return spec, pr, regs, peaks, report, matches


def write_outputs(base, spec, pr, regs, peaks, report, args):
    if args.png or args.outdir:
        from .plotting import plot_spectrum
        out = args.png if (args.png and not args.outdir) else base + ".png"
        plot_spectrum(pr, regs, peaks, path=out)
    if args.csv or args.outdir:
        out = args.csv if (args.csv and not args.outdir) else base + "_peaks.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ppm", "Hz", "height", "note"])
            for p in sorted(peaks, key=lambda p: -p.ppm):
                w.writerow([f"{p.ppm:.4f}", f"{p.ppm * spec.obs:.2f}", f"{p.height:.6g}", p.note])
        if regs:
            with open(os.path.splitext(out)[0].replace("_peaks", "") + "_integrals.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["from_ppm", "to_ppm", "center", "nH", "mult", "J_Hz", "raw", "tag", "include"])
                for r in regs:
                    w.writerow([f"{r.start:.4f}", f"{r.stop:.4f}", f"{r.center:.4f}", f"{r.nH:.3f}", r.mult,
                                " ".join(f"{j:.1f}" for j in r.J), f"{r.value:.6g}", r.tag, int(r.include)])
    if args.spectrum_csv or args.outdir:
        out = args.spectrum_csv if (args.spectrum_csv and not args.outdir) else base + "_spectrum.csv"
        import numpy as np
        np.savetxt(out, np.c_[pr.ppm, pr.real], delimiter=",", header="ppm,intensity", comments="", fmt="%.6f,%.6g")
    if args.outdir:
        with open(base + "_report.txt", "w") as fh:
            fh.write(report + "\n")
            if matches:
                fh.write(A.impurity_summary(matches) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jebru-cli", description="JEOL/Bruker 1D NMR processing & analysis")
    ap.add_argument("paths", nargs="+", help=".jdf / Bruker experiment folder / fid / .zip")
    ap.add_argument("--lb", type=float, default=-1, help="line broadening in Hz (default: 1H 0.3, 13C 1.0)")
    ap.add_argument("--zf", type=int, default=2, help="zero-filling factor")
    ap.add_argument("--p0", type=float, help="zero-order phase (disables auto phasing)")
    ap.add_argument("--p1", type=float, help="first-order phase")
    ap.add_argument("--solvent", help="solvent (CDCl3, DMSO-d6, CD3OD, D2O, ...)")
    ap.add_argument("--no-ref", action="store_true", help="do not reference to the solvent signal")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--snr", type=float, default=6.0, help="peak detection S/N threshold")
    ap.add_argument("--png", help="save figure (png/pdf/svg)")
    ap.add_argument("--csv", help="peak list CSV")
    ap.add_argument("--spectrum-csv", help="spectrum data CSV (ppm,intensity)")
    ap.add_argument("--outdir", help="batch output folder (figure, CSVs and report per file)")
    args = ap.parse_args(argv)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    rc = 0
    for path in args.paths:
        try:
            spec, pr, regs, peaks, report, matches = run_one(path, args)
        except Exception as e:  # keep going in batch mode
            print(f"[error] {path}: {e}", file=sys.stderr)
            rc = 1
            continue
        name = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
        if len(args.paths) > 1 or args.outdir:
            print(f"# {path}")
        print(report)
        if matches:
            print("  " + A.impurity_summary(matches))
        print(f"  ({spec.vendor}, {spec.nucleus}, {pr.ref_note}, p0={pr.params.p0:.1f}, p1={pr.params.p1:.1f})",
              file=sys.stderr)
        base = os.path.join(args.outdir, name) if args.outdir else name
        write_outputs(base, spec, pr, regs, peaks, report, args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
