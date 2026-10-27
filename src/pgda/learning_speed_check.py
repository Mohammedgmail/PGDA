# 16-learning_speed_check.py — independent check of ARIA's learning-speed study
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# Pulls the training curves straight from W&B and recomputes, per seed:
#   - normalised AUC of the success curve over 0-1M ENV steps (not W&B "Step")
#   - success at 100k / 250k / 500k and final (mean of the last 50k)
#   - "sustained" threshold step: first step after which the smoothed curve
#     never drops below 50 / 70 / 80 % again
# and seed-paired tests (t-test + Wilcoxon) between groups.
# It also diffs the run configs, so a code or hyperparameter difference between
# the PGDA and E runs (the main way such a comparison goes wrong) is visible.
#
# Usage:  pip install wandb scipy matplotlib   (wandb login once)
#         fill ENTITY / PROJECT (and the Oracle IDs if you want that group),
#         python 16-learning_speed_check.py
# Outputs: learning_speed_check.json, learning_speed_check.png

import json
import sys
import warnings
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaNs before 1st log

ENTITY = "m-alfandi87-alfandi"
PROJECT = "physically-grounded-diffusion"

# Same run IDs as the evaluation scripts (13 / 15). Pairing is by seed.
GROUPS = {
    "PGDA (B)": {42: "sgl94py3", 123: "1ktap4xi", 456: "lqckfjem",
                 789: "n8cfyu4x", 1011: "289a4luz"},
    "E (coupled, no cue)": {42: "lt56b7lo", 123: "t72zdfyj", 456: "azkfw077",
                            789: "k2nomi95", 1011: "7djx8mih"},
    # Optional, no new training needed: same dynamics as PGDA, perfect cue.
    # Seed 42 is from script 13; fill the rest from the auto-discovered
    # models/<run_id>/ folders, or leave blank to skip the group.
    "Oracle (C)": {42: "y0ruahqi", 123: "", 456: "", 789: "", 1011: ""},
}
COMPARISONS = [("E (coupled, no cue)", "PGDA (B)"),
               ("Oracle (C)", "PGDA (B)"),
               ("E (coupled, no cue)", "Oracle (C)")]

METRICS = ["success/rate", "rollout/success_rate"]   # tried in this order
STEP_KEYS = ["global_step", "time/total_timesteps", "train/global_step",
             "num_timesteps"]                         # must be ENV steps
TOTAL = 1_000_000
DX = 5_000                    # grid spacing (env steps)
SMOOTH_STEPS = 25_000         # trailing moving average for "sustained"
THRESHOLDS = [0.5, 0.7, 0.8]
CHECKPOINTS = [100_000, 250_000, 500_000]
CONFIG_IGNORE = ("seed", "mode", "name", "id", "path", "dir", "_wandb",
                 "randomization", "group", "tags", "notes")
MOCK = False                  # True = synthetic curves (tests this script only)


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch(run_id):
    """Return (env_steps, success, metric, step_key, config) for one run."""
    import wandb
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/{run_id}")
    rows = list(run.scan_history())
    for metric in METRICS:
        for sk in STEP_KEYS:
            pts = [(r[sk], r[metric]) for r in rows
                   if r.get(metric) is not None and r.get(sk) is not None]
            if len(pts) > 20:
                x, y = np.array(pts, dtype=float).T
                return x, y, metric, sk, dict(run.config)
    keys = sorted({k for r in rows[:500] for k in r})
    sys.exit(f"\nRun {run_id}: no success metric logged together with an env-step "
             f"key.\nKeys present: {keys}\nSend me this list and I will adapt "
             f"METRICS / STEP_KEYS.")


def fetch_mock(run_id, group, seed):
    rng = np.random.default_rng(abs(hash((group, seed))) % 2**32)
    x = np.sort(rng.uniform(1e4, TOTAL, 4000))
    tau = {"PGDA (B)": 1.6e5, "E (coupled, no cue)": 0.8e5}.get(group, 1.2e5)
    y = np.clip(0.86 * (1 - np.exp(-x / (tau * rng.uniform(0.85, 1.15))))
                + rng.normal(0, 0.05, x.size), 0, 1)
    return x, y, "mock", "mock", {"learning_rate": 3e-4, "ent_coef": 0.1,
                                  "seed": seed, "randomization_mode": group}


# ── Analysis ──────────────────────────────────────────────────────────────────

def on_grid(x, y, grid):
    order = np.argsort(x)
    x, y = x[order], y[order]
    x, idx = np.unique(x, return_index=True)
    return np.interp(grid, x, y[idx], left=np.nan, right=np.nan)


def trailing_mean(v, w):
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - w + 1): i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def sustained(grid, smooth, thr):
    ok = smooth >= thr
    ok[np.isnan(smooth)] = False
    for i in range(len(ok)):
        if ok[i:].all():
            return int(grid[i])
    return None


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def paired(name_a, name_b, per_seed, field):
    a, b = per_seed[name_a], per_seed[name_b]
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3:
        return None
    da = np.array([a[s][field] for s in seeds])
    db = np.array([b[s][field] for s in seeds])
    d = da - db
    t = stats.ttest_rel(da, db)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    try:
        w = stats.wilcoxon(da, db).pvalue
    except ValueError:
        w = float("nan")
    return {"seeds": seeds, "mean_diff": float(d.mean()),
            "ci95": [float(ci[0]), float(ci[1])], "t_p": float(t.pvalue),
            "wilcoxon_p": float(w), "wins": f"{int((d > 0).sum())}/{len(d)}",
            "per_seed_diff": [float(v) for v in d]}


def main():
    if not MOCK and not (ENTITY and PROJECT):
        sys.exit("Fill ENTITY and PROJECT at the top of the script first.")
    grid = np.arange(0, TOTAL + DX, DX)
    w = max(1, SMOOTH_STEPS // DX)
    curves, per_seed, configs, used = {}, {}, {}, set()

    for g, ids in GROUPS.items():
        for seed, rid in ids.items():
            if not rid:
                continue
            x, y, metric, sk, cfg = (fetch_mock(rid, g, seed) if MOCK
                                     else fetch(rid))
            used.add((metric, sk))
            v = on_grid(x, y, grid)
            curves.setdefault(g, {})[seed] = v
            configs[f"{g} | seed {seed} | {rid}"] = flatten(cfg)
            print(f"  {g:22s} seed {seed:<5d} {rid}: {x.size} points, "
                  f"env steps {x.min():,.0f}-{x.max():,.0f}")
            if x.max() < 0.5 * TOTAL:
                print(f"    WARNING: max step {x.max():,.0f} — '{sk}' does not "
                      f"look like env steps.")

    print(f"\nMetric / step key used: {sorted(used)}")
    if len(used) > 1:
        print("WARNING: runs used different metric/step keys — not comparable.")

    # Common window: from the latest first log to the earliest last log
    all_v = [v for g in curves.values() for v in g.values()]
    valid = np.all([~np.isnan(v) for v in all_v], axis=0)
    lo, hi = grid[valid].min(), grid[valid].max()
    print(f"Common window for AUC: {lo:,} - {hi:,} env steps")
    win = (grid >= lo) & (grid <= hi)

    for g, seeds in curves.items():
        per_seed[g] = {}
        for seed, v in seeds.items():
            sm = trailing_mean(v, w)
            rec = {"auc": float(np.nanmean(v[win])),
                   "final": float(np.nanmean(v[(grid > hi - 50_000) & win]))}
            for c in CHECKPOINTS:
                rec[f"at_{c // 1000}k"] = float(sm[grid == c][0])
            for thr in THRESHOLDS:
                rec[f"sustained_{int(thr * 100)}"] = sustained(grid, sm, thr)
            per_seed[g][seed] = rec

    # Report
    print("\n" + "=" * 78)
    for g, seeds in per_seed.items():
        print(f"\n{g}  (n={len(seeds)} seeds)")
        for f in ["auc", "final"] + [f"at_{c // 1000}k" for c in CHECKPOINTS]:
            vals = np.array([r[f] for r in seeds.values()])
            print(f"  {f:14s} {vals.mean():.3f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.3f}"
                  f"   per seed: {np.round(vals, 3).tolist()}")
        for thr in THRESHOLDS:
            vals = [r[f"sustained_{int(thr * 100)}"] for r in seeds.values()]
            txt = ["never" if v is None else f"{v // 1000}k" for v in vals]
            print(f"  sustained {int(thr * 100)}%  per seed: {txt}")
        mean_curve = trailing_mean(np.nanmean(np.stack(list(curves[g].values())), 0), w)
        print("  sustained (mean curve): " + ", ".join(
            f"{int(t * 100)}% at " + (lambda s: "never" if s is None else f"{s // 1000}k")(
                sustained(grid, mean_curve, t)) for t in THRESHOLDS))

    tests = {}
    print("\nSeed-paired comparisons (A − B):")
    for a, b in COMPARISONS:
        if a not in per_seed or b not in per_seed:
            continue
        for f in ["auc", "at_100k", "final"]:
            r = paired(a, b, per_seed, f)
            if r is None:
                continue
            tests[f"{a} vs {b} | {f}"] = r
            print(f"  {a} − {b} [{f:8s}] {r['mean_diff']:+.3f} "
                  f"(95% CI {r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}), "
                  f"t-test p={r['t_p']:.4f}, Wilcoxon p={r['wilcoxon_p']:.4f}, "
                  f"A higher in {r['wins']}")

    # Config diff
    keys = sorted({k for c in configs.values() for k in c})
    diffs = {}
    for k in keys:
        if any(s in k.lower() for s in CONFIG_IGNORE):
            continue
        vals = {n: repr(c.get(k, "<missing>")) for n, c in configs.items()}
        if len(set(vals.values())) > 1:
            diffs[k] = vals
    print("\nConfig keys that differ between runs (seed/mode/name ignored):")
    if not diffs:
        print("  none — the runs share all logged hyperparameters.")
    for k, vals in diffs.items():
        print(f"  {k}:")
        for n, v in vals.items():
            print(f"      {n}: {v}")

    with open("learning_speed_check.json", "w") as f:
        json.dump({"metric_step_keys": sorted(used),
                   "window": [int(lo), int(hi)], "per_seed": {
                       g: {str(s): r for s, r in d.items()} for g, d in per_seed.items()},
                   "tests": tests, "config_diffs": diffs}, f, indent=2)
    print("\nSaved -> learning_speed_check.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for g, seeds in curves.items():
            M = np.stack([trailing_mean(v, w) for v in seeds.values()])
            m = np.nanmean(M, 0)
            se = np.nanstd(M, 0, ddof=1) / np.sqrt(M.shape[0]) if M.shape[0] > 1 else 0 * m
            line, = ax.plot(grid / 1e3, m, label=f"{g} (n={M.shape[0]})")
            ax.fill_between(grid / 1e3, m - se, m + se, alpha=0.2,
                            color=line.get_color())
        ax.set_xlabel("Environment steps (k)")
        ax.set_ylabel("Training success rate")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Mean ± s.e. across seeds", fontsize=9)
        fig.tight_layout()
        fig.savefig("learning_speed_check.png", dpi=200)
        print("Saved -> learning_speed_check.png")
    except Exception as e:                      # plotting is optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
