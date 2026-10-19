"""R3-C11: exact empirical capacity sensitivity of the submitted band-mean statistic.

Run from any directory: E2/.venv/Scripts/python.exe E2/scripts/run_sensitivity.py
No fitting, interpolation, resampling, new reference simulations, or GPU training.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
FIG = HERE / "figures"
O5 = ROOT / "6-code" / "O5-fema-p58"
CFG = O5 / "configs" / "o5_equipment_period_bands_supported.json"
ANCHORS = [0.4, 1.0, 2.0]
LABELS = ["suitable", "conditional", "unsuitable"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def category(p):
    return np.where(p < 0.1, 0, np.where(p < 0.5, 1, 2))


def probability(sorted_values, c, side="left"):
    return (sorted_values.size - np.searchsorted(sorted_values, c, side=side)) / sorted_values.size


def merged_intervals(edges, ref_class, fno_class):
    """For >= exceedance, q is constant on (edge[i], edge[i+1]]."""
    pair = ref_class * 3 + fno_class
    cuts = np.r_[0, np.flatnonzero(np.diff(pair)) + 1, len(pair)]
    result = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        r, p = int(ref_class[start]), int(fno_class[start])
        result.append({"capacity_left_g": float(edges[start]), "capacity_right_g": float(edges[stop]),
                       "left_closed": False, "right_closed": True,
                       "reference_category": LABELS[r], "fno_category": LABELS[p],
                       "agree": r == p, "nonconservative": p < r,
                       "width_g": float(edges[stop] - edges[start])})
    return result


def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{stem}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    bands = list(cfg["period_bands"].items())
    anchor_rows, summary_rows, intervals_rows, boundary_rows, curves = [], [], [], [], []
    env = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
           "executable": sys.executable, "platform": platform.platform(),
           "packages": {p: importlib.metadata.version(p) for p in ["numpy", "matplotlib"]},
           "statistic": "mean_{period in band} mean_{sample} I(Sa>=c)",
           "boundaries": "suitable q<0.10; conditional 0.10<=q<0.50; unsuitable q>=0.50",
           "capacity_domain": "[0, max(reference,FNO spectral demands in each band)]",
           "input_files": {str(CFG.relative_to(ROOT)): sha256(CFG)},
           "script_sha256": sha256(__file__), "data": {}}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 5.3), sharey=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for row, sid in enumerate(["F10", "FW10"]):
        path = O5 / "results" / "multistructure" / "frs_full" / f"{sid}_roof_frs_5pct.npz"
        env["input_files"][str(path.relative_to(ROOT))] = sha256(path)
        src = np.load(path)
        periods = src["periods_s"].astype(np.float64)
        ref, pred = src["fem_sa_g"][:, 0, :], src["fno_sa_g"][:, 0, :]
        assert ref.shape == pred.shape == (27018, 90)
        assert np.all(np.isfinite(ref)) and np.all(np.isfinite(pred))
        assert np.all(ref >= 0) and np.all(pred >= 0)
        env["data"][sid] = {"n_scaled_samples": ref.shape[0], "n_periods": len(periods),
                             "periods_s": periods.tolist(), "bands": {}}
        for col, (bid, b) in enumerate(bands):
            mask = (periods >= b["period_min_s"]) & (periods <= b["period_max_s"])
            r, p = ref[:, mask].astype(np.float64).ravel(), pred[:, mask].astype(np.float64).ravel()
            n = r.size
            sr, sp = np.sort(r), np.sort(p)
            edges = np.unique(np.r_[0., sr, sp])
            qr, qp = probability(sr, edges[:-1], "right"), probability(sp, edges[:-1], "right")
            cr, cp = category(qr), category(qp)
            intervals = merged_intervals(edges, cr, cp)
            for segment in intervals:
                intervals_rows.append({"structure": sid, "band": bid, **segment})
            mismatch_width = sum(x["width_g"] for x in intervals if not x["agree"])
            nonconservative_width = sum(x["width_g"] for x in intervals if x["nonconservative"])
            worst = int(np.argmax(np.abs(qp - qr)))
            summary_rows.append({"structure": sid, "band": bid, "n_periods_in_band": int(mask.sum()),
                "n_sample_period_pairs": n, "empirical_max_capacity_g": float(edges[-1]),
                "max_abs_exceedance_gap_pp": float(100 * abs(qp[worst] - qr[worst])),
                "gap_signed_fno_minus_ref_pp": float(100 * (qp[worst] - qr[worst])),
                "gap_capacity_interval_left_g": float(edges[worst]), "gap_capacity_interval_right_g": float(edges[worst+1]),
                "classification_disagreement_width_g": mismatch_width,
                "nonconservative_disagreement_width_g": nonconservative_width,
                "agreement_capacity_length_percent": 100*(1-mismatch_width/edges[-1]),
                "integrated_false_negative_pair_rate_g": float(np.maximum(r-p, 0).mean()),
                "integrated_false_positive_pair_rate_g": float(np.maximum(p-r, 0).mean())})
            for c in ANCHORS:
                re, pe = r >= c, p >= c
                a, z = float(re.mean()), float(pe.mean())
                # Independent reproduction of the manuscript's order of operations.
                direct_r = float((ref[:, mask].astype(np.float64) >= c).mean(axis=0).mean())
                assert np.isclose(a, direct_r, rtol=0, atol=1e-14)
                fn, fp = int(np.sum(re & ~pe)), int(np.sum(~re & pe))
                anchor_rows.append({"structure": sid, "band": bid, "capacity_g": c,
                    "reference_percent": 100*a, "fno_percent": 100*z, "gap_pp": 100*(z-a),
                    "reference_category": LABELS[int(category(a))], "fno_category": LABELS[int(category(z))],
                    "n_pairs": n, "false_negative_pairs": fn, "false_positive_pairs": fp,
                    "false_negative_percent_all_pairs": 100*fn/n,
                    "false_negative_percent_reference_exceedances": 100*fn/int(re.sum()) if re.sum() else 0,
                    "false_positive_percent_all_pairs": 100*fp/n,
                    "paired_exceedance_agreement_percent": 100*(1-(fn+fp)/n)})
                boundary_rows.append({"structure": sid, "band": bid, "capacity_g": c,
                    "boundary_difference_interval_left_percent": 100*min(a,z),
                    "boundary_difference_interval_right_percent": 100*max(a,z),
                    "width_pp": 100*abs(a-z),
                    "lower_boundary_convention": "Suitable q<l: split when min(q)<l<=max(q)",
                    "upper_boundary_convention": "Unsuitable q>=u: split when min(q)<u<=max(q)",
                    "direction": "FNO less restrictive" if z<a else ("FNO more restrictive" if z>a else "identical")})
            # Preserve every empirical change point in a compact lossless numerical artifact.
            np.savez_compressed(OUT / f"{sid}_{bid}_exact_empirical_curves.npz", capacity_edges_g=edges,
                                ref_q_on_left_open_right_closed_intervals=qr,
                                fno_q_on_left_open_right_closed_intervals=qp)
            # Dense visualization only; all reported metrics/intervals above use exact data.
            class_edges = [seg["capacity_left_g"] for seg in intervals] + [edges[-1]]
            x = np.unique(np.r_[np.linspace(0,edges[-1],3001), ANCHORS, edges[worst]])
            yr, yp = probability(sr,x), probability(sp,x)
            curves += [{"structure":sid,"band":bid,"capacity_g":float(c),"reference_percent":100*float(a),
                        "fno_percent":100*float(z)} for c,a,z in zip(x,yr,yp)]
            ax = axes[row,col]
            ax.plot(x,100*yr,color="#0072B2",label="Reference MDOF-NTHA",lw=1.8)
            ax.plot(x,100*yp,color="#D55E00",label="FNO",lw=1.6)
            for c in ANCHORS:
                ax.axvline(c,color="0.75",ls=":",lw=.8)
            for y in [10,50]: ax.axhline(y,color="0.5",ls="--",lw=.8)
            ax.set_title(f"{sid.replace('10','-10')}: {b['period_min_s']:g}-{b['period_max_s']:g} s")
            ax.set_xlim(0,edges[-1]); ax.set_ylim(0,100)
            ax.set_xlabel("Demand threshold c (g)")
            if col == 0: ax.set_ylabel("Band-mean exceedance (%)")
            ax.grid(alpha=.18)
            ax2 = axes2[row,col]
            for seg in intervals:
                color = "#F3C184" if seg["nonconservative"] else ("#DADADA" if seg["agree"] else "#B3D3E8")
                ax2.axvspan(seg["capacity_left_g"],seg["capacity_right_g"],color=color,zorder=0)
            xc = np.asarray(class_edges)
            ax2.step(xc,category(probability(sr,xc,"right")),where="post",color="#0072B2",label="Reference",lw=1.8)
            ax2.step(xc,category(probability(sp,xc,"right")),where="post",color="#D55E00",label="FNO",lw=1.5,ls="--")
            ax2.set_title(ax.get_title()); ax2.set_xlim(0,edges[-1]); ax2.set_ylim(-.15,2.15)
            ax2.set_yticks([0,1,2],LABELS); ax2.set_xlabel("Demand threshold c (g)")
            env["data"][sid]["bands"][bid] = {"n_period_points":int(mask.sum()),"period_points_s":periods[mask].tolist(),
                                             "n_exact_capacity_intervals":len(qr)}
            print(f"{sid} {bid}: max gap {summary_rows[-1]['max_abs_exceedance_gap_pp']:.3f} pp; "
                  f"disagreement intervals {[ (s['capacity_left_g'],s['capacity_right_g']) for s in intervals if not s['agree']]}",flush=True)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle("Capacity sensitivity: full empirical demand range; 27,018 samples per structure",fontsize=12)
    fig.tight_layout(); savefig(fig,"capacity_exceedance_sensitivity")
    axes2[0,0].legend(fontsize=8,frameon=False)
    fig2.suptitle("Study-defined 10% / 50% classes; orange = FNO less restrictive",fontsize=12)
    fig2.tight_layout(); savefig(fig2,"classification_capacity_intervals")
    write_csv("anchor_results.csv",anchor_rows)
    write_csv("table11_reproduction.csv",[a for a in anchor_rows if a["capacity_g"]==1])
    write_csv("exact_sensitivity_summary.csv",summary_rows)
    write_csv("classification_capacity_intervals.csv",intervals_rows)
    write_csv("classification_boundary_sensitivity.csv",boundary_rows)
    write_csv("capacity_curve_plot_grid.csv",curves)
    old_csv = O5 / "results" / "multistructure" / "chapter6_revision" / "o5_period_band_exceedance_summary.csv"
    old = list(csv.DictReader(old_csv.open(encoding="utf-8-sig")))
    comparison=[]
    for a in anchor_rows:
        if a["capacity_g"] != 1: continue
        for source,key in [("FEM","reference_percent"),("FNO","fno_percent")]:
            match=next(z for z in old if z["structure_id"]==a["structure"] and z["band_id"]==a["band"] and z["source"]==source)
            error=a[key]-float(match["mean_exceedance_percent"])
            assert abs(error)<1e-10, (a,match,error)
            comparison.append({"structure":a["structure"],"band":a["band"],"source":source,"recomputed_percent":a[key],
                               "archived_percent":match["mean_exceedance_percent"],"difference_pp":error})
    write_csv("archived_table11_validation.csv",comparison)
    env["input_files"][str(old_csv.relative_to(ROOT))]=sha256(old_csv)
    audit_inputs = [O5 / "scripts" / "plot_chapter6_revision_figures.py",
                    O5 / "scripts" / "run_equipment_placement_risk.py",
                    O5 / "scripts" / "compute_floor_response_spectra.py"]
    audit_inputs += list((O5 / "resources").rglob("BD-3.9.5*.pdf"))
    audit_inputs += list((O5 / "resources").rglob("BD-3.9.10*.pdf"))
    audit_inputs += [O5 / "results" / "multistructure" / "frs_full" / f"{sid}_frs_summary.json" for sid in ["F10","FW10"]]
    audit_inputs += [O5 / "results" / "multistructure" / "demands" / "o5_demand_export_summary.json",
                     O5 / "configs" / "o5_multistructure_cases.json"]
    for path in audit_inputs: env["input_files"][str(path.relative_to(ROOT))] = sha256(path)
    (OUT / "environment_input_manifest.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    print("Table 11 matched archived CSV within 1e-10 percentage points. Outputs complete.")


if __name__ == "__main__": main()
