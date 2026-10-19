"""Independent direct-count checks of interval endpoints and archived agreement."""
import csv
import json
from pathlib import Path
import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
O5 = ROOT / "6-code" / "O5-fema-p58"
cfg = json.loads((O5 / "configs" / "o5_equipment_period_bands_supported.json").read_text())
labels = ["suitable", "conditional", "unsuitable"]
def classify(p): return labels[0 if p < .1 else 1 if p < .5 else 2]
rows = list(csv.DictReader((HERE / "results" / "classification_capacity_intervals.csv").open()))
checks = []
for sid in ["F10", "FW10"]:
    d = np.load(O5 / "results" / "multistructure" / "frs_full" / f"{sid}_roof_frs_5pct.npz")
    for bid,b in cfg["period_bands"].items():
        mask = (d["periods_s"] >= b["period_min_s"]) & (d["periods_s"] <= b["period_max_s"])
        r,p = d["fem_sa_g"][:,0,mask].astype(float), d["fno_sa_g"][:,0,mask].astype(float)
        for row in [x for x in rows if x["structure"]==sid and x["band"]==bid]:
            left,right = float(row["capacity_left_g"]),float(row["capacity_right_g"])
            for c in [np.nextafter(left, np.inf), (left+right)/2, right]:
                rc, pc = classify(np.mean(r >= c)), classify(np.mean(p >= c))
                assert (rc,pc)==(row["reference_category"],row["fno_category"]), (row,c,rc,pc)
                checks.append({"structure":sid,"band":bid,"capacity":c,"reference":rc,"fno":pc})
        assert classify(np.mean(r>=0))==classify(np.mean(p>=0))=="unsuitable"
for stem in ["capacity_exceedance_sensitivity", "classification_capacity_intervals"]:
    pdf = PdfReader(HERE / "figures" / f"{stem}.pdf")
    assert len(pdf.pages)==1
    assert "F-10" in pdf.pages[0].extract_text()
print(f"PASS: {len(checks)} independent direct-count endpoint/midpoint class checks; c=0 endpoints; two 1-page PDF text checks.")
(HERE / "results" / "verification_summary.json").write_text(json.dumps({"passed":True,"direct_count_checks":len(checks),
    "pdf_single_page_text_checked":2,"table11_reference_validation":"archived_table11_validation.csv",
    "figures_visually_checked":False},indent=2))
