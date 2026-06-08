# -*- coding: utf-8 -*-
"""
Analyse the plan-components ablation.

Reads every raw output CSV in OUTPUT_DIR/raw, then produces:
  scored/<model>_<variant>_scored.csv     per-sample scores + parsed verdicts + metrics
  summary/variant_summary.csv              accuracy / invalid / SSC / add-drop per variant
  summary/per_task_accuracy.csv            accuracy by StatQA task per variant
  summary/pairwise_tests.csv               McNemar + paired-bootstrap (within model)
  summary/label_clarity_stratified.csv     accuracy/SSC/over-sel by Step-4 clarity (+CA trend)
  summary/cross_capacity.csv               14B vs 72B on shared variants (V0/V1/V2)
  annotation/<model>_<variant>_label_clarity_sheet.csv   50-sample human sheets (V0-V3, 14B)
  summary/decision_report.txt              maps results onto design 5.4.5 decision logic

Run AFTER run_ablation.py:
  python analyze_ablation.py
"""
import os
import glob
import math
import argparse
import numpy as np
import pandas as pd

import config
import statqa_core as core
import parsing as parse


# ==========================================================================
# 1. Score one raw file
# ==========================================================================
def score_file(raw_path):
    df = pd.read_csv(raw_path)
    model_tag, variant = os.path.basename(raw_path).replace("_raw.csv", "").split("_", 1)

    rows = []
    for _, r in df.iterrows():
        gt_m = core.gt_methods(r["results"])
        gt_c = core.gt_columns(r["relevant_column"])
        pred = core.extract_final_answer(r.get("model_output", ""))
        sc = core.score_selection(pred, gt_c, gt_m)

        rec = {
            "row_id": r["row_id"], "task": r["task"], "difficulty": r.get("difficulty", ""),
            "model": model_tag, "variant": variant,
            "valid": sc["valid"],
            "overall_correct": sc["overall_correct"],
            "methods_correct": sc["methods_correct"],
            "columns_correct": sc["columns_correct"],
            "over_selection": int(sc["m_wrong"] > 0),
            "under_selection": int(sc["m_missed"] > 0),
            "n_pred_methods": len(pred["methods"]) if pred else 0,
            "n_gt_methods": len(gt_m),
            "pred_methods": "|".join(pred["methods"]) if pred else "",
            "gt_methods": "|".join(gt_m),
        }

        # Step-4 verdict metrics (only where Step 4 exists)
        if variant in config.VARIANTS_WITH_STEP4:
            verdicts, region = parse.parse_verdicts(r.get("model_output", ""), variant)
            final_methods = pred["methods"] if pred else []
            m = parse.compute_ssc_metrics(verdicts, final_methods)
            rec.update(m)
            rec["label_clarity_auto"] = parse.label_clarity_auto(verdicts, region)
            rec["verdicts"] = "|".join(f"{k}={v}" for k, v in verdicts.items())
        else:
            for k in ("ssc", "drop_rate", "add_rate_in", "add_rate_out"):
                rec[k] = math.nan
            rec["label_clarity_auto"] = "n/a"
            rec["verdicts"] = ""
        rows.append(rec)

    scored = pd.DataFrame(rows)

    # V4 self-check change rate (compare Turn-1 vs final selection)
    if variant == "V4":
        changes = []
        for _, r in df.iterrows():
            t1 = core.extract_final_answer(r.get("turn1_output", ""))
            t2 = core.extract_final_answer(r.get("model_output", ""))
            if t1 is None or t2 is None:
                changes.append(np.nan)
            else:
                s1 = {x.lower().strip() for x in t1["methods"]}
                s2 = {x.lower().strip() for x in t2["methods"]}
                changes.append(int(s1 != s2))
        scored["selfcheck_changed"] = changes

    os.makedirs(os.path.join(config.OUTPUT_DIR, "scored"), exist_ok=True)
    scored.to_csv(config.scored_path(model_tag, variant), index=False)
    return scored


# ==========================================================================
# 2. Statistics helpers
# ==========================================================================
def mcnemar_test(a, b):
    """a, b: paired 0/1 arrays (1=correct). Returns (b01, b10, p_value)."""
    a, b = np.asarray(a), np.asarray(b)
    b01 = int(np.sum((a == 1) & (b == 0)))   # a correct, b wrong
    b10 = int(np.sum((a == 0) & (b == 1)))   # a wrong, b correct
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[0, b01], [b10, 0]], exact=(b01 + b10) < 25)
        p = res.pvalue
    except Exception:
        from scipy.stats import binomtest
        n = b01 + b10
        p = binomtest(min(b01, b10), n, 0.5).pvalue if n > 0 else 1.0
    return b01, b10, p


def paired_bootstrap_diff(a, b, iters=None, ci=None, seed=config.SEED):
    """Mean(a)-Mean(b) with bootstrap CI over paired samples."""
    iters = iters or config.BOOTSTRAP_ITERS
    ci = ci or config.CI
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    point = a.mean() - b.mean()
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return point, lo, hi


def cochran_armitage_trend(success, total, scores):
    """
    Cochran-Armitage trend test across ordered groups.
    success/total/scores: equal-length sequences (one entry per ordered group).
    Returns (z, p_two_sided).
    """
    from scipy.stats import norm
    x = np.asarray(success, float)
    n = np.asarray(total, float)
    t = np.asarray(scores, float)
    N = n.sum()
    X = x.sum()
    if N == 0 or X == 0 or X == N:
        return float("nan"), float("nan")
    p = X / N
    num = np.sum(t * (x - n * p))
    var = p * (1 - p) * (np.sum(n * t ** 2) - (np.sum(n * t) ** 2) / N)
    if var <= 0:
        return float("nan"), float("nan")
    z = num / math.sqrt(var)
    pval = 2 * (1 - norm.cdf(abs(z)))
    return z, pval


def _mean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s) else float("nan")


def _nanmean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmean(s)) if len(s.dropna()) else float("nan")


# ==========================================================================
# 3. Summary tables
# ==========================================================================
def variant_summary(all_scored):
    rows = []
    for (model, variant), g in all_scored.groupby(["model", "variant"]):
        valid = g[g["valid"] == 1]
        incorrect = g[g["overall_correct"] == 0]
        rows.append({
            "model": model, "variant": variant, "n": len(g),
            "overall_accuracy": _mean(g["overall_correct"]),
            "methods_accuracy": _mean(g["methods_correct"]),
            "columns_accuracy": _mean(g["columns_correct"]),
            "invalid_rate": 1 - _mean(g["valid"]),
            "over_selection_rate": _mean(g["over_selection"]),
            "under_selection_rate": _mean(g["under_selection"]),
            "over_sel_rate_among_errors": _mean(incorrect["over_selection"]) if len(incorrect) else float("nan"),
            "mean_ssc": _nanmean(g.get("ssc", pd.Series(dtype=float))),
            "mean_drop_rate": _nanmean(g.get("drop_rate", pd.Series(dtype=float))),
            "mean_add_rate_in": _nanmean(g.get("add_rate_in", pd.Series(dtype=float))),
            "mean_add_rate_out": _nanmean(g.get("add_rate_out", pd.Series(dtype=float))),
            "mean_pred_methods": _mean(g["n_pred_methods"]),
            "selfcheck_change_rate": _nanmean(g["selfcheck_changed"]) if "selfcheck_changed" in g else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["model", "variant"])


def per_task_accuracy(all_scored):
    t = (all_scored.groupby(["model", "variant", "task"])["overall_correct"]
         .mean().reset_index().rename(columns={"overall_correct": "overall_accuracy"}))
    return t.pivot_table(index=["model", "variant"], columns="task",
                         values="overall_accuracy").reset_index()


def pairwise_tests(all_scored):
    """McNemar + paired bootstrap on overall_correct, within each model, on shared rows."""
    pairs = [("V0", "V1"), ("V0", "V2"), ("V0", "V3"),
             ("V2", "V3"), ("V2", "V4"), ("V3", "V4"), ("V0", "V4")]
    rows = []
    for model, g in all_scored.groupby("model"):
        wide = g.pivot_table(index="row_id", columns="variant", values="overall_correct")
        for x, y in pairs:
            if x not in wide or y not in wide:
                continue
            sub = wide[[x, y]].dropna()
            if sub.empty:
                continue
            a, b = sub[x].values, sub[y].values
            b01, b10, p = mcnemar_test(a, b)
            diff, lo, hi = paired_bootstrap_diff(a, b)
            rows.append({
                "model": model, "comparison": f"{x} vs {y}", "n_paired": len(sub),
                "acc_x": float(a.mean()), "acc_y": float(b.mean()),
                "acc_diff(x-y)": diff, "ci_lo": lo, "ci_hi": hi,
                "mcnemar_b(x>y)": b01, "mcnemar_b(y>x)": b10, "mcnemar_p": p,
            })
    return pd.DataFrame(rows)


def label_clarity_stratified(all_scored):
    """For 14B V0-V3: accuracy / over-selection / SSC by Step-4 label clarity (auto proxy)."""
    order = ["narrative_only", "partially_labeled", "fully_labeled"]
    score_map = {c: i for i, c in enumerate(order)}
    rows, trend_rows = [], []
    sub = all_scored[(all_scored["model"] == "14B") &
                     (all_scored["variant"].isin(config.LABEL_CLARITY_VARIANTS))]
    for variant, g in sub.groupby("variant"):
        succ, tot, scrs = [], [], []
        for clarity in order:
            gg = g[g["label_clarity_auto"] == clarity]
            if len(gg) == 0:
                continue
            rows.append({
                "model": "14B", "variant": variant, "label_clarity_auto": clarity,
                "n": len(gg),
                "overall_accuracy": _mean(gg["overall_correct"]),
                "over_selection_rate": _mean(gg["over_selection"]),
                "mean_ssc": _nanmean(gg["ssc"]),
            })
            succ.append(int(gg["overall_correct"].sum()))
            tot.append(len(gg))
            scrs.append(score_map[clarity])
        if len(tot) >= 2:
            z, p = cochran_armitage_trend(succ, tot, scrs)
            trend_rows.append({"model": "14B", "variant": variant,
                               "ca_trend_z": z, "ca_trend_p": p,
                               "groups": ">".join([order[s] for s in scrs])})
    return pd.DataFrame(rows), pd.DataFrame(trend_rows)


def cross_capacity(all_scored):
    """14B vs 72B on shared variants (V0/V1/V2): accuracy + SSC, paired bootstrap on shared rows."""
    rows = []
    # 72B may not be in config (e.g. running 14B only). If absent, or if no 72B
    # rows were scored, skip cross-capacity entirely and return an empty frame.
    if "72B" not in config.MODELS or all_scored[all_scored["model"] == "72B"].empty:
        return pd.DataFrame(rows)
    shared_variants = config.MODELS["72B"]["variants"]
    for variant in shared_variants:
        g14 = all_scored[(all_scored["model"] == "14B") & (all_scored["variant"] == variant)]
        g72 = all_scored[(all_scored["model"] == "72B") & (all_scored["variant"] == variant)]
        if g14.empty or g72.empty:
            continue
        merged = pd.merge(g14[["row_id", "overall_correct", "ssc"]],
                          g72[["row_id", "overall_correct", "ssc"]],
                          on="row_id", suffixes=("_14B", "_72B")).dropna(subset=["overall_correct_14B", "overall_correct_72B"])
        if merged.empty:
            continue
        diff, lo, hi = paired_bootstrap_diff(merged["overall_correct_72B"].values,
                                             merged["overall_correct_14B"].values)
        rows.append({
            "variant": variant, "n_paired": len(merged),
            "acc_14B": _mean(merged["overall_correct_14B"]),
            "acc_72B": _mean(merged["overall_correct_72B"]),
            "acc_diff(72B-14B)": diff, "ci_lo": lo, "ci_hi": hi,
            "ssc_14B": _nanmean(merged["ssc_14B"]),
            "ssc_72B": _nanmean(merged["ssc_72B"]),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 4. Human annotation sheets (50 samples per variant, 14B, V0-V3)
# ==========================================================================
def export_annotation_sheets(raw_dir):
    os.makedirs(os.path.join(config.OUTPUT_DIR, "annotation"), exist_ok=True)
    for variant in config.LABEL_CLARITY_VARIANTS:
        raw = config.raw_path("14B", variant)
        if not os.path.exists(raw):
            continue
        df = pd.read_csv(raw)
        n = min(config.N_ANNOTATION_SAMPLES, len(df))
        sample = df.sample(n=n, random_state=config.ANNOTATION_SEED).copy()
        out = []
        for _, r in sample.iterrows():
            pred = core.extract_final_answer(r.get("model_output", ""))
            verdicts, region = parse.parse_verdicts(r.get("model_output", ""), variant)
            out.append({
                "row_id": r["row_id"], "task": r["task"],
                "question": r["refined_question"],
                "gt_methods": "|".join(core.gt_methods(r["results"])),
                "pred_methods": "|".join(pred["methods"]) if pred else "",
                "auto_label_clarity": parse.label_clarity_auto(verdicts, region),
                "auto_verdicts": "|".join(f"{k}={v}" for k, v in verdicts.items()),
                # ---- columns for the human annotator to fill ----
                "HUMAN_label_clarity[fully_labeled/partially_labeled/narrative_only]": "",
                "HUMAN_notes": "",
                "model_output": r.get("model_output", ""),
            })
        pd.DataFrame(out).to_csv(config.annotation_path("14B", variant), index=False)
        print(f"[save] annotation sheet: {config.annotation_path('14B', variant)}")


# ==========================================================================
# 5. Decision report (design 5.4.5)
# ==========================================================================
def write_decision_report(vsum, tests, strat, trend, cross, path):
    def acc(model, v):
        r = vsum[(vsum.model == model) & (vsum.variant == v)]
        return float(r["overall_accuracy"].iloc[0]) if len(r) else float("nan")

    lines = []
    L = lines.append
    L("PLAN-COMPONENTS ABLATION -- DECISION REPORT (design 5.4.5)\n" + "=" * 60)
    L("\n[14B variant overall accuracy]")
    for v in config.ALL_VARIANTS:
        L(f"  {v}: {acc('14B', v):.4f}")

    L("\n[Q1 intervenability: do forced structured verdicts (V2/V3) beat V0?]")
    for cmp in ("V0 vs V2", "V0 vs V3", "V2 vs V3"):
        r = tests[(tests.model == "14B") & (tests.comparison == cmp)]
        if len(r):
            r = r.iloc[0]
            L(f"  {cmp}: diff(x-y)={r['acc_diff(x-y)']:+.4f} "
              f"CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] McNemar p={r['mcnemar_p']:.4g}")
    L("  [decision rules, design 5.4.5 -- apply to the numbers above]")
    L("  IF V2/V3 >> V0 (CI excludes 0): H1a is interventable; narrative reasoning is")
    L("     the main form of judgement-generation failure.")
    L("  IF V3 >> V2: narrative 'reason' itself induces over-inclusion.")
    L("  IF V2 ~ V3 (both > V0): what matters is being forced to give an explicit verdict,")
    L("     not the presence/absence of the reason.")

    L("\n[Q2 self-check value: V1 (no Step 5) vs V0]")
    r = tests[(tests.model == "14B") & (tests.comparison == "V0 vs V1")]
    if len(r):
        r = r.iloc[0]
        L(f"  V0 vs V1: diff={r['acc_diff(x-y)']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
          f"McNemar p={r['mcnemar_p']:.4g}")
    L("  [decision rule] IF V1 ~ V0: self-check is a no-op corrective component (echoes 4.3.3).")

    L("\n[Lower-bound control: V4 (post-hoc self-check, no Step 4) vs V2/V3]")
    for cmp in ("V2 vs V4", "V3 vs V4"):
        r = tests[(tests.model == "14B") & (tests.comparison == cmp)]
        if len(r):
            r = r.iloc[0]
            L(f"  {cmp}: diff(x-y)={r['acc_diff(x-y)']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['mcnemar_p']:.4g}")
    L("  [decision rule] IF V4 << V2/V3: H1a mitigation needs structure AT GENERATION TIME;")
    L("     post-hoc self-check alone cannot repair over-inclusion.")

    L("\n[Core analysis: accuracy by Step-4 label clarity + Cochran-Armitage trend]")
    if len(trend):
        for _, r in trend.iterrows():
            L(f"  {r['variant']}: CA trend z={r['ca_trend_z']:.3f} p={r['ca_trend_p']:.4g}  ({r['groups']})")
    L("  [decision rule] IF accuracy rises monotonically with clarity (CA trend p<.05) and")
    L("     over-selection is highest in narrative_only: directly corroborates H1a")
    L("     (over-selection leaks from un-verdicted methods).")
    L("  [decision rule] IF accuracy in the fully_labeled subset is still well below 1.0:")
    L("     H1a refines to a CONTENT failure (the verdict itself is over-inclusive), not")
    L("     merely a FORM failure (verdict never generated). See design 5.4.4.")

    L("\n[Cross-capacity (14B vs 72B, V0/V1/V2)]")
    if len(cross):
        for _, r in cross.iterrows():
            L(f"  {r['variant']}: acc 14B={r['acc_14B']:.4f} 72B={r['acc_72B']:.4f} "
              f"diff(72B-14B)={r['acc_diff(72B-14B)']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")

    L("\n[Fallback]")
    L("  If NO variant (incl. V2/V3) significantly beats V0 => H1a not fixable by plan-form")
    L("  intervention; points to a deeper parametric method prior (Shapiro-Wilk / Bartlett /")
    L("  F-Test etc.), to be treated as an alternative explanation in the Discussion.")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("[save] decision report:", path)
    print("\n".join(lines))


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(config.OUTPUT_DIR, "raw"))
    args = ap.parse_args()

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "*_raw.csv")))
    if not raw_files:
        raise SystemExit(f"[!] no raw files in {args.raw_dir}; run run_ablation.py first.")

    os.makedirs(os.path.join(config.OUTPUT_DIR, "summary"), exist_ok=True)
    all_scored = pd.concat([score_file(p) for p in raw_files], ignore_index=True)
    print(f"[i] scored {len(all_scored)} rows from {len(raw_files)} files")

    vsum = variant_summary(all_scored)
    ptask = per_task_accuracy(all_scored)
    tests = pairwise_tests(all_scored)
    strat, trend = label_clarity_stratified(all_scored)
    cross = cross_capacity(all_scored)

    sdir = os.path.join(config.OUTPUT_DIR, "summary")
    vsum.to_csv(os.path.join(sdir, "variant_summary.csv"), index=False)
    ptask.to_csv(os.path.join(sdir, "per_task_accuracy.csv"), index=False)
    tests.to_csv(os.path.join(sdir, "pairwise_tests.csv"), index=False)
    strat.to_csv(os.path.join(sdir, "label_clarity_stratified.csv"), index=False)
    trend.to_csv(os.path.join(sdir, "label_clarity_trend.csv"), index=False)
    cross.to_csv(os.path.join(sdir, "cross_capacity.csv"), index=False)

    export_annotation_sheets(args.raw_dir)
    write_decision_report(vsum, tests, strat, trend, cross,
                          os.path.join(sdir, "decision_report.txt"))

    print("\n[done] all summaries written to", sdir)


if __name__ == "__main__":
    main()