import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def log_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def save_metrics_table(records, tex_path):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in records for k in r.keys()})
    lines = [r"\begin{tabular}{%s}" % ("l" * len(keys)), r"\toprule", " & ".join(keys) + r"\\ \midrule"]
    for r in records:
        lines.append(" & ".join(str(r.get(k, "")) for k in keys) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines), encoding="utf-8")

def save_fig_phase(results, img_path):
    ds = {}
    for r in results:
        ds.setdefault(r["rho"], {})[r["delta"]] = r["success"]
    rhos = sorted(ds.keys())
    deltas = sorted({d for m in ds.values() for d in m.keys()})
    Z = np.array([[ds[r].get(d, 0.0) for d in deltas] for r in rhos])
    plt.figure()
    plt.imshow(Z, origin="lower", aspect="auto",
               extent=[min(deltas), max(deltas), min(rhos), max(rhos)])
    plt.xlabel(r"$\delta = M/N$"); plt.ylabel(r"$\rho = K/N$")
    plt.title("Success Probability")
    plt.colorbar()
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=180, bbox_inches="tight")
    plt.close()
