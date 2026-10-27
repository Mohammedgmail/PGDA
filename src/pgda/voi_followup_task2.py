# 21b-voi_followup_task2.py — Task-2 pilot gate: follow-up analysis (no simulation)
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# Reads task2_voi_results.json written by 21-scripted_voi_task2.py and adds two
# analyses to the pre-registered gate that script 21 prints.
#
# A. DESIGN A AS RUN (feedback after contact allowed), all valid episodes
#    Script 21's gate optimises the probing policy's feature thresholds and
#    continuations in-sample. With few episodes that overstates S_probe, so
#    here every strategy is also chosen on one half of the episodes and scored
#    on the other (repeated, stratified 2-fold cross-validation). Supplementary
#    only: the pre-registered decision for Design A is script 21's gate.
#    Also reported: how much the cue adds to a policy that can probe as well
#    (cue + probing vs probing alone, in-sample).
#
# B. COMMITTED STRIKE (pre-registered on 13 Oct 2026, before episodes 120-959
#    existed). Variant of Design A: at the first control step with contact or
#    motion, the arm keeps that step's command for K more steps and then
#    retracts, so nothing learned from the contact can change the push. The
#    policy must commit before contact (requirement R3). The scripted rollouts
#    of script 21 already contain these pushes: direct (v, n = K + 1).
#      Design, chosen on episodes 0-119 with script 21's selection rule
#      (largest worst-case margin; ties -> delta = 5 cm, then smaller K):
#          K = 2 (3 push steps), D uniform in [6, 10] cm, delta = 5 cm
#      Confirmation on episodes 120-959 only (not used for the choice):
#          S_cat - S_single >= 15 pp  and  S_cat >= 60 % for every material,
#          computed exactly as script 21 computes its gate.
#      Reported with it: bootstrap 95% CI (episodes resampled within material,
#      strategies re-optimised) and a cross-validated gap.
#
# PRE-REGISTERED ORDER OF DECISION
#   1. Script 21's gate passes for Design A on all 960 episodes -> Design A
#      (feedback allowed) with the design script 21 chooses.
#   2. Otherwise, the committed strike passes on episodes 120-959 -> committed
#      strike, K = 2, D in [6, 10] cm, delta = 5 cm.
#   3. Otherwise neither: report both and move to Design B (or report the
#      negative result).
#   Everything is reported whichever way it goes.
#
# RUN:  python 21b-voi_followup_task2.py      (seconds to a few minutes)
#
# VERSION 2 (13 Oct 2026): section A now treats episodes in which a probe
#   never touched the object as their own bin, as script 21 does (version 1
#   mis-scored them, which understated S_probe in section A only). Section B
#   and the decision are unchanged.
# OUT:  printed summary and task2_voi_followup.json

import json
import itertools
import os
import sys

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_FILE  = "task2_voi_results.json"
OUT_FILE      = "task2_voi_followup.json"
N_FULL        = 960                       # episodes of the full pilot run
SELECTION_K   = (0, 120)                  # episodes used to choose the committed design
CONFIRM_K     = (120, 960)                # episodes used only to confirm it
COMMITTED     = {"K": 2, "band": (0.06, 0.10), "delta": 0.05}   # pre-registered
GATE_GAP_PP, GATE_MIN_PCT = 15.0, 60.0
N_BOOT, N_CV  = 1000, 200
CV_TOP_A      = 5                         # Design A designs to cross-validate
SEED          = 20261013

CATEGORIES = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]
BANDS  = [(0.06, 0.10), (0.08, 0.12), (0.10, 0.14), (0.12, 0.16), (0.14, 0.18),
          (0.16, 0.20), (0.18, 0.22), (0.20, 0.24), (0.08, 0.15), (0.10, 0.20)]
DELTAS = [0.03, 0.04, 0.05]
SPLIT_Q = np.arange(0.1, 1.0, 0.1)


# ── Data ──────────────────────────────────────────────────────────────────────

def k3(x):
    return f"{x:.3f}"


class Data:
    def __init__(self, res):
        cfg = res["config"]
        self.V, self.NMAX, self.PV = cfg["v_grid"], cfg["n_max"], cfg["probe_v"]
        eps = sorted([e for e in res["episodes"].values() if e.get("valid")], key=lambda e: e["k"])
        self.k = np.array([e["k"] for e in eps])
        self.cat = np.array([e["category"] for e in eps])
        self.opts = ([("d", None, v, n - 1) for v in self.V for n in range(1, self.NMAX + 1)]
                     + [("p", ap, "stop", 0) for ap in self.PV]
                     + [("p", ap, v2, j) for ap in self.PV for v2 in self.V
                        for j in range(1, self.NMAX)])
        E, O = len(eps), len(self.opts)
        self.S, self.L = np.full((E, O), np.nan), np.full((E, O), np.nan)
        self.OK = np.zeros((E, O), bool)
        for i, e in enumerate(eps):
            for o, (f, ap, v, j) in enumerate(self.opts):
                if f == "d":
                    r = e["direct"].get(f"{k3(v)}|{j + 1}")
                elif v == "stop":
                    r = e["probe"].get(f"{k3(ap)}|stop|0")
                else:
                    r = e["probe"].get(f"{k3(ap)}|{k3(v)}|{j}")
                if r:
                    self.S[i, o], self.L[i, o] = r[0], r[1]
                    self.OK[i, o] = (r[2] & 1) == 1 and (r[2] & 2) == 0
        self.J = np.array([o[3] for o in self.opts])
        self.DIR = np.array([o[0] == "d" for o in self.opts])
        self.PROBE = {ap: np.array([o[0] == "p" and o[1] == ap for o in self.opts]) for ap in self.PV}
        self.F = {ap: np.array([e["feature"].get(k3(ap), [np.nan])[0] for e in eps]) for ap in self.PV}
        self._c = {}

    def succ(self, D, delta):
        key = (round(D, 4), delta)
        if key not in self._c:
            with np.errstate(invalid="ignore"):
                self._c[key] = self.OK & (((self.S - D) ** 2 + self.L ** 2) < delta ** 2)
        return self._c[key]


def d_values(band):
    return [round(x, 2) for x in np.arange(band[0], band[1] + 1e-9, 0.01)]


def stratified_halves(rng, cat, ix):
    A, B = [], []
    for c in CATEGORIES:
        idx = rng.permutation(ix[cat[ix] == c])
        h = len(idx) // 2
        A += list(idx[:h])
        B += list(idx[h:])
    return np.array(A), np.array(B)


# ── Committed strike ──────────────────────────────────────────────────────────

def committed(data, ix, K, band, delta):
    sel = data.DIR & (data.J == K)
    acc = {k: [] for k in ["single", "cat"] + [f"cat_{c}" for c in CATEGORIES]}
    for D in d_values(band):
        s = data.succ(D, delta)[ix][:, sel]
        acc["single"].append(s.mean(0).max())
        tot = 0.0
        for c in CATEGORIES:
            m = data.cat[ix] == c
            v = s[m].mean(0).max() if m.any() else 0.0
            acc[f"cat_{c}"].append(v)
            tot += m.sum() * v
        acc["cat"].append(tot / len(ix))
    r = {k: 100 * float(np.mean(v)) for k, v in acc.items()}
    r["gap"] = r["cat"] - r["single"]
    r["min_cat"] = min(r[f"cat_{c}"] for c in CATEGORIES)
    r["passes"] = bool(r["gap"] >= GATE_GAP_PP and r["min_cat"] >= GATE_MIN_PCT)
    r["score"] = min(r["gap"] - GATE_GAP_PP, r["min_cat"] - GATE_MIN_PCT)
    return r


def committed_cv(data, ix, K, band, delta, rng):
    sel = np.where(data.DIR & (data.J == K))[0]
    gaps, mins = [], []
    for _ in range(N_CV):
        A, B = stratified_halves(rng, data.cat, ix)
        for tr, te in ((A, B), (B, A)):
            single, catv, per = [], [], {c: [] for c in CATEGORIES}
            for D in d_values(band):
                s = data.succ(D, delta)[:, sel]
                single.append(s[te][:, np.argmax(s[tr].mean(0))].mean())
                tot = 0.0
                for c in CATEGORIES:
                    mtr, mte = tr[data.cat[tr] == c], te[data.cat[te] == c]
                    v = s[mte][:, np.argmax(s[mtr].mean(0))].mean()
                    per[c].append(v)
                    tot += len(mte) * v
                catv.append(tot / len(te))
            gaps.append(100 * (np.mean(catv) - np.mean(single)))
            mins.append(100 * min(np.mean(v) for v in per.values()))
    return {"gap_mean": float(np.mean(gaps)), "gap_2.5": float(np.percentile(gaps, 2.5)),
            "gap_97.5": float(np.percentile(gaps, 97.5)), "min_cat_mean": float(np.mean(mins))}


def committed_boot(data, ix, K, band, delta, rng):
    g = []
    for _ in range(N_BOOT):
        b = np.concatenate([rng.choice(ix[data.cat[ix] == c], (data.cat[ix] == c).sum())
                            for c in CATEGORIES])
        g.append(committed(data, b, K, band, delta)["gap"])
    return [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]


def speed_table(data, ix, K, band, delta):
    """success (%) by commanded speed and material, averaged over D in the band"""
    sel = np.where(data.DIR & (data.J == K))[0]
    out = {}
    for c in CATEGORIES:
        m = ix[data.cat[ix] == c]
        out[c] = [100 * float(np.mean([data.succ(D, delta)[m][:, o].mean() for D in d_values(band)]))
                  for o in sel]
    return [data.opts[o][2] for o in sel], out


# ── Design A (feedback allowed) ───────────────────────────────────────────────

def _bins(f, P, fit=True, policy=None):
    """Probing policy on the probe feature f: up to 3 bins, one continuation per
    bin; episodes in which the probe never touched the object (f is NaN) get a
    continuation of their own, exactly as in script 21.
    fit=True : returns (total successes, policy) of the best policy on (f, P).
    fit=False: returns the success rate of `policy` on (f, P)."""
    P = P.astype(float)
    nan = np.isnan(f)
    if not fit:
        cuts, choice, nan_choice = policy
        idx = np.empty(len(f), int)
        b = np.searchsorted(np.array(cuts), f[~nan], side="left") if cuts else np.zeros((~nan).sum(), int)
        idx[~nan] = np.array(choice)[b]
        idx[nan] = nan_choice
        return P[np.arange(len(f)), idx].mean()
    nan_choice = int(np.argmax(P[nan].sum(0) if nan.any() else P.sum(0)))
    base = P[nan][:, nan_choice].sum() if nan.any() else 0.0
    f, P = f[~nan], P[~nan]
    if len(f) == 0:
        return base, ([], [nan_choice], nan_choice)
    order = np.argsort(f)
    fs, Ps = f[order], P[order]
    cum = np.vstack([np.zeros(P.shape[1]), np.cumsum(Ps, 0)])

    def seg(a, b):
        t = cum[b] - cum[a]
        return t.max(), int(np.argmax(t))
    cand = sorted({int(np.searchsorted(fs, x, side="right")) for x in np.quantile(f, SPLIT_Q)}
                  - {0, len(f)})
    v0, o0 = seg(0, len(f))
    best = (v0, [], [o0])
    for c in cand:
        (a1, o1), (a2, o2) = seg(0, c), seg(c, len(f))
        if a1 + a2 > best[0]:
            best = (a1 + a2, [fs[c - 1]], [o1, o2])
    for c1, c2 in itertools.combinations(cand, 2):
        (a1, o1), (a2, o2), (a3, o3) = seg(0, c1), seg(c1, c2), seg(c2, len(f))
        if a1 + a2 + a3 > best[0]:
            best = (a1 + a2 + a3, [fs[c1 - 1], fs[c2 - 1]], [o1, o2, o3])
    return best[0] + base, (best[1], best[2], nan_choice)


def design_a(data, ix, K, band, delta):
    """in-sample, as script 21 (plus cue + probing)"""
    allowed = data.J <= K
    acc = {k: [] for k in ["single", "probe", "cat", "catprobe"]}
    for D in d_values(band):
        s = data.succ(D, delta)[ix]
        sa = s[:, allowed]
        acc["single"].append(s[:, allowed & data.DIR].mean(0).max())
        acc["probe"].append(max(_bins(data.F[ap][ix], s[:, allowed & data.PROBE[ap]])[0]
                                for ap in data.PV) / len(ix))
        tot_c = tot_cp = 0.0
        for c in CATEGORIES:
            m = data.cat[ix] == c
            best = sa[m].sum(0).max()
            tot_c += best
            for ap in data.PV:
                best = max(best, _bins(data.F[ap][ix][m], s[m][:, allowed & data.PROBE[ap]])[0])
            tot_cp += best
        acc["cat"].append(tot_c / len(ix))
        acc["catprobe"].append(tot_cp / len(ix))
    r = {k: 100 * float(np.mean(v)) for k, v in acc.items()}
    r["gap"] = r["cat"] - max(r["single"], r["probe"])
    return r


def design_a_cv(data, ix, K, band, delta, rng, reps):
    allowed = data.J <= K
    dsel = np.where(allowed & data.DIR)[0]
    rows = []
    for _ in range(reps):
        A, B = stratified_halves(rng, data.cat, ix)
        for tr, te in ((A, B), (B, A)):
            sg, pg, cg = [], [], []
            for D in d_values(band):
                s = data.succ(D, delta)
                sa = s[:, allowed]
                sg.append(s[te][:, dsel[np.argmax(s[tr][:, dsel].mean(0))]].mean())
                tot = 0.0
                for c in CATEGORIES:
                    mtr, mte = tr[data.cat[tr] == c], te[data.cat[te] == c]
                    tot += len(mte) * sa[mte][:, np.argmax(sa[mtr].mean(0))].mean()
                cg.append(tot / len(te))
                best = None
                for ap in data.PV:
                    psel = np.where(allowed & data.PROBE[ap])[0]
                    v, pol = _bins(data.F[ap][tr], s[tr][:, psel])
                    if best is None or v > best[0]:
                        best = (v, ap, psel, pol)
                _, ap, psel, pol = best
                pg.append(_bins(data.F[ap][te], s[te][:, psel], fit=False, policy=pol))
            rows.append((np.mean(sg), np.mean(pg), np.mean(cg)))
    r = 100 * np.array(rows)
    gap = r[:, 2] - np.maximum(r[:, 0], r[:, 1])
    return {"single": float(r[:, 0].mean()), "probe": float(r[:, 1].mean()),
            "cat": float(r[:, 2].mean()), "gap_mean": float(gap.mean()),
            "gap_2.5": float(np.percentile(gap, 2.5)), "gap_97.5": float(np.percentile(gap, 97.5))}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        sys.exit(f"{RESULTS_FILE} not found: run 21-scripted_voi_task2.py first.")
    with open(RESULTS_FILE) as f:
        res = json.load(f)
    data = Data(res)
    rng = np.random.default_rng(SEED)
    n_all = len(res["episodes"])
    all_ix = np.arange(len(data.k))
    sel_ix = np.where((data.k >= SELECTION_K[0]) & (data.k < SELECTION_K[1]))[0]
    con_ix = np.where((data.k >= CONFIRM_K[0]) & (data.k < CONFIRM_K[1]))[0]
    count = lambda ix: {c: int((data.cat[ix] == c).sum()) for c in CATEGORIES}
    out = {"n_episodes": n_all, "n_valid": int(len(all_ix)), "counts_all": count(all_ix),
           "counts_selection": count(sel_ix), "counts_confirmation": count(con_ix),
           "committed_design": {**COMMITTED, "band": list(COMMITTED["band"])}}
    print(f"{'#' * 96}\n  TASK-2 PILOT FOLLOW-UP | {len(all_ix)} valid of {n_all} episodes | "
          f"selection {len(sel_ix)} ({count(sel_ix)}), confirmation {len(con_ix)} ({count(con_ix)})\n{'#' * 96}")
    if n_all < N_FULL:
        print(f"  NOTE: only {n_all} of {N_FULL} episodes: the confirmation set is incomplete or empty.")

    # A. Design A as run
    a_rows = []
    for K, band, delta in itertools.product(range(data.NMAX), BANDS, DELTAS):
        a_rows.append(((K, band, delta), design_a(data, all_ix, K, band, delta)))
    a_rows.sort(key=lambda x: -x[1]["gap"])
    a21 = res.get("analysis", {})
    print(f"\n  A. DESIGN A (feedback allowed), all {len(all_ix)} episodes")
    print(f"     script 21's pre-registered gate: {a21.get('n_passing', '?')} design(s) pass "
          f"(analysis over {a21.get('n_valid', '?')} episodes)")
    print(f"     {'K':>2} {'band':>9} {'delta':>5} | in-sample: {'single':>6} {'probe':>6} {'cue':>6} "
          f"{'gap':>6} {'cue+probe':>9} | cross-validated: {'probe':>6} {'cue':>6} {'gap':>6} [95% range]")
    a_out = []
    for (K, band, delta), r in a_rows[:CV_TOP_A]:
        cv = design_a_cv(data, all_ix, K, band, delta, rng, reps=max(20, N_CV // 4))
        a_out.append({"K": K, "band": list(band), "delta": delta, "in_sample": r, "cv": cv})
        print(f"     {K:2d} {band[0] * 100:4.0f}-{band[1] * 100:<4.0f} {delta * 100:5.0f} | "
              f"{r['single']:16.1f} {r['probe']:6.1f} {r['cat']:6.1f} {r['gap']:+6.1f} {r['catprobe']:9.1f} | "
              f"{cv['probe']:22.1f} {cv['cat']:6.1f} {cv['gap_mean']:+6.1f} "
              f"[{cv['gap_2.5']:+.1f}, {cv['gap_97.5']:+.1f}]")
    print("     cue+probe: best policy that knows the material AND can probe (in-sample)")
    out["design_a_top"] = a_out

    # B. Committed strike
    K, band, delta = COMMITTED["K"], COMMITTED["band"], COMMITTED["delta"]
    print(f"\n  B. COMMITTED STRIKE (pre-registered K = {K}, D {band[0] * 100:.0f}-{band[1] * 100:.0f} cm, "
          f"delta = {delta * 100:.0f} cm)")
    sel_rows = sorted(((d, committed(data, sel_ix, *d)) for d in
                       itertools.product(range(data.NMAX), BANDS, DELTAS)),
                      key=lambda x: (-x[1]["score"], x[0][2] != 0.05, x[0][0]))
    chosen = sel_rows[0][0]
    print(f"     selection rule on episodes {SELECTION_K[0]}-{SELECTION_K[1] - 1} picks: K = {chosen[0]}, "
          f"D {chosen[1][0] * 100:.0f}-{chosen[1][1] * 100:.0f} cm, delta = {chosen[2] * 100:.0f} cm "
          f"({'matches' if chosen == (K, band, delta) else 'DIFFERS FROM'} the pre-registered design)")
    out["selection_rule_pick"] = {"K": chosen[0], "band": list(chosen[1]), "delta": chosen[2]}
    for name, ix in (("selection", sel_ix), ("confirmation", con_ix)):
        if len(ix) < 30:
            print(f"     {name}: too few episodes ({len(ix)})")
            continue
        r = committed(data, ix, K, band, delta)
        boot = committed_boot(data, ix, K, band, delta, rng)
        cv = committed_cv(data, ix, K, band, delta, rng)
        out[f"committed_{name}"] = {"gate": r, "boot_ci95": boot, "cv": cv}
        print(f"     {name:12s} ({len(ix):3d} ep.): single {r['single']:5.1f}  cue {r['cat']:5.1f}  "
              + "  ".join(f"{c.split()[-1]} {r['cat_' + c]:5.1f}" for c in CATEGORIES)
              + f"  | gap {r['gap']:+5.1f} [boot {boot[0]:+.1f}, {boot[1]:+.1f}]  "
              f"CV gap {cv['gap_mean']:+.1f}, CV min material {cv['min_cat_mean']:.1f}  "
              f"-> {'PASS' if r['passes'] else 'fail'}")
    ix = con_ix if len(con_ix) >= 30 else sel_ix
    speeds, tab = speed_table(data, ix, K, band, delta)
    print(f"     success (%) by commanded speed, {'confirmation' if ix is con_ix else 'selection'} episodes, "
          f"averaged over D:")
    print("       v (m/s)          " + "".join(f"{v:7.2f}" for v in speeds))
    for c in CATEGORIES:
        print(f"       {c:17s}" + "".join(f"{x:7.0f}" for x in tab[c]))
    out["committed_speed_table"] = {"speeds": speeds, "success": tab}

    # Decision
    a_pass = a21.get("n_passing", 0) > 0 and a21.get("n_valid", 0) >= N_FULL * 0.95
    c_res = out.get("committed_confirmation", {}).get("gate")
    print(f"\n{'#' * 96}\n  PRE-REGISTERED DECISION")
    if a21.get("n_valid", 0) < N_FULL * 0.95 or c_res is None:
        decision = "incomplete: finish the full run (QUICK = False in script 21), then re-run this script"
    elif a_pass:
        decision = "Design A (feedback allowed), with the design script 21 chose"
    elif c_res["passes"]:
        decision = f"committed strike, K = {K}, D {band[0] * 100:.0f}-{band[1] * 100:.0f} cm, delta = {delta * 100:.0f} cm"
    else:
        decision = "neither design passes: report both, move to Design B or report the negative result"
    print(f"  -> {decision}\n{'#' * 96}")
    out["decision"] = decision
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  Saved -> {OUT_FILE}")