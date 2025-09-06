#!/usr/bin/env python3
import argparse, json, time, os, yaml, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- FIX: Add project root to path before other imports ---
# The project root is 3 levels up from this script file.
# (run_experiment.py -> runners -> experiments -> project_root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Your library
from ampire import *
#from ampire.core.kamp import KAMP          # KF-AMP
#from ampire.distributed.distributed_kamp import DistributedKAMP  # DKF-AMP
#from ampire.core.enet_convex_hull import EnetConvexHull            # Elastic-net OC
#from ampire.utils.metrics import nmse

# Experiment loaders/reporting
from experiments.datasets.synthetic import make_gaussian_cs
from experiments.datasets.images import load_image_set
from experiments.datasets.odds import load_odds_dataset
from experiments.datasets.intel_lab import load_intel_timeseries
from experiments.metrics.reporting import save_metrics_table, save_fig_phase, log_jsonl

def nmse(x_true, x_est):
    """Normalized Mean Squared Error"""
    return np.mean((x_true - x_est.flatten())**2) / np.mean(x_true**2)

def run_synthetic_phase(cfg, outdir):
    grid = cfg["grid"]     # {'delta': [...], 'rho': [...], 'trials': 50, 'snr_db': 40}
    n      = cfg.get("n", 2000)
    lam1   = cfg.get("lambda1", 1e-2)
    lam2   = cfg.get("lambda2", 0.2)
    algo   = cfg.get("algo", "KF-AMP")  # 'AMP' | 'KF-AMP' | 'DKF-AMP'
    results = []
    errors = {}

    for d in tqdm(grid["delta"], desc="Delta"):
        for r in tqdm(grid["rho"], desc="Rho"):
            succ = 0; nmse_sum = 0.0; t_sum = 0.0; trial_count = 0
            for trial_idx in tqdm(range(grid["trials"]), desc="Trials"):
                A, x0, y, sigma = make_gaussian_cs(n=n, delta=d, rho=r, snr_db=grid["snr_db"])
                t0 = time.time()
                try:
                    if algo == "KF-AMP":
                        # Adjust to your actual API if needed:
                        kamp = KAMP(alpha=0.5, tau=lam1, max_iter=100)
                        est = kamp.fit(A, y).solve()
                    elif algo == "DKF-AMP":
                        est = DistributedKAMP.from_matrix(A, y).fit().x_hat
                    else:
                        # Simple baseline AMP via KF-AMP with Q=0 (if your code supports it)
                        kamp = KAMP(alpha=0.5, tau=lam1, max_iter=100)
                        kamp.fit(A, y)
                        kamp.Q = 0.0  # Set Q to 0 for baseline
                        est = kamp.solve()
                    t_sum += (time.time() - t0)
                    e = nmse(x0, est)
                    nmse_sum += e
                    succ += int(e < 1e-5)
                    trial_count += 1
                except Exception as e:
                    errors[f"{d}_{r}_{trial_idx}"] = str(e)
                    continue
            if trial_count > 0:
                results.append({
                    "delta": d, "rho": r,
                    "nmse": nmse_sum / trial_count,
                    "success": succ / trial_count,
                    "time": t_sum / trial_count
                })
            else:
                results.append({
                    "delta": d, "rho": r,
                    "nmse": float('inf'),
                    "success": 0.0,
                    "time": 0.0
                })
            log_jsonl(results, outdir / "synthetic_phase.jsonl")
    save_fig_phase(results, outdir / "phase_heatmap.png")
    save_metrics_table(results, outdir / "phase_table.tex")
    with open(outdir / "errors.json", "w") as f:
        json.dump(errors, f)
    return results

def run_images_cs(cfg, outdir):
    ds = cfg["dataset"]    # 'set12' | 'bsd68' | 'kodak24'
    mratio = cfg.get("m_over_n", 0.25)
    lam1   = cfg.get("lambda1", 1e-3)
    lam2   = cfg.get("lambda2", 0.2)

    images = load_image_set(ds)
    metrics = []
    for name, x0 in tqdm(images, desc="Images"):
        A, y, meta = make_gaussian_cs(n=x0.size, delta=mratio, rho=None,
                                      return_signal=False, x_img=x0)
        kamp = KAMP(alpha=0.5, tau=lam1, max_iter=100)
        est = kamp.fit(A, y).solve()
        rec = {
            "name": name,
            "psnr": meta["psnr_fn"](x0, est.reshape(x0.shape)),
            "ssim": meta["ssim_fn"](x0, est.reshape(x0.shape))
        }
        metrics.append(rec)
    log_jsonl(metrics, outdir / f"{ds}_cs.jsonl")
    save_metrics_table(metrics, outdir / f"{ds}_cs.tex")
    return metrics

def run_distributed_sensor(cfg, outdir):
    topo = cfg["topology"]   # 'dag' | 'cycle' | 'selfloop' | 'mixed'
    data = load_intel_timeseries()  # {node_id: (A_i, y_i)}
    dk = DistributedKAMP.from_partition(data, topology=topo, **cfg.get("dkamp", {})).fit()
    report = dk.report()     # should include consensus_error, nmse_global, bytes, iters
    log_jsonl([report], outdir / f"distributed_{topo}.jsonl")
    save_metrics_table([report], outdir / f"distributed_{topo}.tex")
    return report

def run_odds_oc(cfg, outdir):
    ds = cfg["dataset"]      # 'kddcup99' | 'arrhythmia' | 'secom' ...
    lam1 = cfg.get("lambda1", 1e-2)
    lam2 = cfg.get("lambda2", 0.2)
    X, y = load_odds_dataset(ds)
    model = EnetConvexHull(l1=lam1, l2=(1-lam2))
    model.fit(X[y==0])       # train on normal samples only
    scores = model.score_samples(X)
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    ap  = average_precision_score(y, -scores)  # lower score → more anomalous
    auc = roc_auc_score(y, -scores)
    f1  = f1_score(y, (scores < np.percentile(scores[y==0], 1)).astype(int))
    res = {"dataset": ds, "AP": ap, "ROC-AUC": auc, "F1@1%": f1}
    log_jsonl([res], outdir / f"odds_{ds}.jsonl")
    save_metrics_table([res], outdir / f"odds_{ds}.tex")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    outdir = Path(args.out) / Path(args.config).stem
    outdir.mkdir(parents=True, exist_ok=True)

    task = cfg["task"]
    if task == "synthetic_phase": run_synthetic_phase(cfg, outdir)
    elif task == "image_cs":      run_images_cs(cfg, outdir)
    elif task == "distributed":   run_distributed_sensor(cfg, outdir)
    elif task == "odds_oc":       run_odds_oc(cfg, outdir)
    else: raise ValueError("Unknown task")

if __name__ == "__main__":
    main()
