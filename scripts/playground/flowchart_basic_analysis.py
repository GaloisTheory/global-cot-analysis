"""Basic Analysis — pn408 rollouts.

Descriptive statistics on CoT trace lengths and condition breakdowns,
loaded directly from rollout JSON files.

Run as a VS-Code / PyCharm "percent" notebook or from the terminal:
    cd projects/global-cot-analysis
    uv run python scripts/playground/flowchart_basic_analysis.py
"""

# %% [markdown]
# # Basic Analysis — pn408
# Load rollouts directly and compute:
# 1. Faithful vs unfaithful counts in cued traces
# 2. CoT trace lengths (sentences & word count) by condition

# %% Cell 1 — Imports & config
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]

PN = "pn408"          # <-- switch here
# PN = "pn37"
# PN = "pn277"
# PN = "pn339"
# PN = "pn408_s100"

ROLLOUTS_DIR = (
    REPO_ROOT
    / "prompts"
    / f"faith_combined_{PN}"
    / "qwen3-8b"
    / "rollouts"
)

PLOTS_DIR = REPO_ROOT / "scripts" / "playground" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"Rollouts dir: {ROLLOUTS_DIR} (exists: {ROLLOUTS_DIR.exists()})")
print(f"Plots dir: {PLOTS_DIR}")

# %% Cell 2 — Load rollouts & build records


def rollout_bucket(resp: dict) -> str:
    """Classify a response into one of three condition buckets."""
    condition = resp.get("condition", "?")
    if condition == "uncued":
        return "uncued"
    if condition == "cued":
        return "cued-unfaithful" if bool(resp.get("unfaithful", False)) else "cued-faithful"
    return f"other:{condition}"


rollout_files = sorted(ROLLOUTS_DIR.glob("*.json"))
assert rollout_files, f"No rollout files found in {ROLLOUTS_DIR}"

records = []
for rpath in rollout_files:
    with open(rpath, "r", encoding="utf-8") as f:
        resp = json.load(f)

    sents = resp.get("sentences", [])
    cot_text = resp.get("cot_content", "")
    word_count = len(cot_text.split()) if cot_text else 0

    records.append({
        "key": f"{resp.get('prompt_index', '?')}_seed{resp.get('seed', '?')}",
        "bucket": rollout_bucket(resp),
        "condition": resp.get("condition", "?"),
        "unfaithful": bool(resp.get("unfaithful", False)),
        "correctness": bool(resp.get("correctness", False)),
        "answer": resp.get("processed_response_content", "?"),
        "n_sentences": len(sents),
        "word_count": word_count,
    })

print(f"Loaded {len(records)} rollouts")

# %% Cell 3 — Condition breakdown (faithful vs unfaithful in cued traces)
BUCKET_ORDER = ["uncued", "cued-faithful", "cued-unfaithful"]
BUCKET_COLORS = {"uncued": "tab:blue", "cued-faithful": "tab:green", "cued-unfaithful": "tab:red"}

bucket_counts = Counter(r["bucket"] for r in records)
bucket_correct = defaultdict(int)
for r in records:
    if r["correctness"]:
        bucket_correct[r["bucket"]] += 1

print("\n" + "=" * 60)
print("CONDITION BREAKDOWN")
print("=" * 60)
print(f"{'Bucket':<20s} {'Count':>6s} {'Correct':>8s} {'Acc %':>7s}")
print("-" * 45)
for b in BUCKET_ORDER:
    n = bucket_counts.get(b, 0)
    c = bucket_correct.get(b, 0)
    acc = 100.0 * c / n if n > 0 else 0.0
    print(f"{b:<20s} {n:6d} {c:8d} {acc:6.1f}%")
total_n = sum(bucket_counts.values())
total_c = sum(bucket_correct.values())
print(f"{'TOTAL':<20s} {total_n:6d} {total_c:8d} {100.0 * total_c / total_n:6.1f}%")

# Within cued only
n_cued = bucket_counts.get("cued-faithful", 0) + bucket_counts.get("cued-unfaithful", 0)
n_unfaithful = bucket_counts.get("cued-unfaithful", 0)
if n_cued > 0:
    print(f"\nCued traces: {n_cued} total, {n_unfaithful} unfaithful ({100.0 * n_unfaithful / n_cued:.1f}%)")

# --- Bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: counts by bucket
ax = axes[0]
counts_ordered = [bucket_counts.get(b, 0) for b in BUCKET_ORDER]
bars = ax.bar(BUCKET_ORDER, counts_ordered, color=[BUCKET_COLORS[b] for b in BUCKET_ORDER], edgecolor="black", linewidth=0.5)
for bar, cnt in zip(bars, counts_ordered):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(cnt),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of traces")
ax.set_title("Traces by Condition")
ax.set_xticks(range(len(BUCKET_ORDER)))
ax.set_xticklabels(BUCKET_ORDER, rotation=15, ha="right")

# Right: correctness rate by bucket
ax = axes[1]
accs = [100.0 * bucket_correct.get(b, 0) / bucket_counts[b] if bucket_counts.get(b, 0) > 0 else 0 for b in BUCKET_ORDER]
bars = ax.bar(BUCKET_ORDER, accs, color=[BUCKET_COLORS[b] for b in BUCKET_ORDER], edgecolor="black", linewidth=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Correctness (%)")
ax.set_title("Correctness Rate by Condition")
ax.set_ylim(0, 105)
ax.set_xticks(range(len(BUCKET_ORDER)))
ax.set_xticklabels(BUCKET_ORDER, rotation=15, ha="right")

plt.suptitle("Condition Breakdown — pn408", fontsize=13)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "01_condition_breakdown.png", dpi=150)
print(f"Saved: {PLOTS_DIR / '01_condition_breakdown.png'}")
plt.close(fig)

# %% Cell 4 — CoT trace length distributions
print("\n" + "=" * 60)
print("COT TRACE LENGTH ANALYSIS")
print("=" * 60)
print("(Word counts are used as a proxy for token counts; no pre-computed token counts in data.)\n")

# Group records by bucket
by_bucket = defaultdict(list)
for r in records:
    by_bucket[r["bucket"]].append(r)


def summarise(values: list, label: str) -> dict:
    arr = np.array(values)
    return {
        "label": label,
        "n": len(arr),
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }


# --- Summary table ---
for metric, field in [("Sentence count", "n_sentences"), ("Word count (~ tokens)", "word_count")]:
    print(f"\n--- {metric} ---")
    print(f"{'Bucket':<20s} {'n':>5s} {'mean':>8s} {'median':>8s} {'std':>8s} {'min':>5s} {'max':>5s}")
    print("-" * 62)
    for b in BUCKET_ORDER:
        vals = [r[field] for r in by_bucket.get(b, [])]
        if not vals:
            continue
        s = summarise(vals, b)
        print(f"{b:<20s} {s['n']:5d} {s['mean']:8.1f} {s['median']:8.1f} {s['std']:8.1f} {s['min']:5.0f} {s['max']:5.0f}")

# --- Statistical tests ---
print("\n--- Statistical Tests (Mann-Whitney U) ---")

# Test 1: uncued vs all-cued
uncued_sents = [r["n_sentences"] for r in by_bucket["uncued"]]
cued_sents = [r["n_sentences"] for r in by_bucket["cued-faithful"]] + [r["n_sentences"] for r in by_bucket["cued-unfaithful"]]
uncued_words = [r["word_count"] for r in by_bucket["uncued"]]
cued_words = [r["word_count"] for r in by_bucket["cued-faithful"]] + [r["word_count"] for r in by_bucket["cued-unfaithful"]]

if uncued_sents and cued_sents:
    u_s, p_s = mannwhitneyu(uncued_sents, cued_sents, alternative="two-sided")
    u_w, p_w = mannwhitneyu(uncued_words, cued_words, alternative="two-sided")
    print(f"  Uncued vs All-Cued (sentences): U={u_s:.0f}, p={p_s:.4g}")
    print(f"  Uncued vs All-Cued (words):     U={u_w:.0f}, p={p_w:.4g}")

# Test 2: cued-faithful vs cued-unfaithful
faith_sents = [r["n_sentences"] for r in by_bucket["cued-faithful"]]
unfaith_sents = [r["n_sentences"] for r in by_bucket["cued-unfaithful"]]
faith_words = [r["word_count"] for r in by_bucket["cued-faithful"]]
unfaith_words = [r["word_count"] for r in by_bucket["cued-unfaithful"]]

if faith_sents and unfaith_sents:
    u_s2, p_s2 = mannwhitneyu(faith_sents, unfaith_sents, alternative="two-sided")
    u_w2, p_w2 = mannwhitneyu(faith_words, unfaith_words, alternative="two-sided")
    print(f"  Faithful vs Unfaithful (sentences): U={u_s2:.0f}, p={p_s2:.4g}")
    print(f"  Faithful vs Unfaithful (words):     U={u_w2:.0f}, p={p_w2:.4g}")

# --- Box plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, field, ylabel in [
    (axes[0], "n_sentences", "Number of sentences"),
    (axes[1], "word_count", "Word count (~ tokens)"),
]:
    data_by_bucket = []
    labels = []
    colors = []
    for b in BUCKET_ORDER:
        vals = [r[field] for r in by_bucket.get(b, [])]
        if vals:
            data_by_bucket.append(vals)
            labels.append(f"{b}\n(n={len(vals)})")
            colors.append(BUCKET_COLORS[b])

    bp = ax.boxplot(data_by_bucket, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")

axes[0].set_title("Sentence Count by Condition")
axes[1].set_title("Word Count by Condition")
plt.suptitle("CoT Trace Length Distributions — pn408", fontsize=13)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "02_trace_lengths_pn408.png", dpi=150)
print(f"Saved: {PLOTS_DIR / '02_trace_lengths_pn408.png'}")
plt.close(fig)

# %% Cell 5 — Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nTotal responses: {len(records)}")
for b in BUCKET_ORDER:
    n = len(by_bucket.get(b, []))
    print(f"  {b}: {n}")

n_cued_total = len(by_bucket.get("cued-faithful", [])) + len(by_bucket.get("cued-unfaithful", []))
n_unfaith = len(by_bucket.get("cued-unfaithful", []))
if n_cued_total > 0:
    print(f"\nCued unfaithful rate: {n_unfaith}/{n_cued_total} = {100.0 * n_unfaith / n_cued_total:.1f}%")

# Median CoT lengths
for b in BUCKET_ORDER:
    rs = by_bucket.get(b, [])
    if rs:
        med_s = np.median([r["n_sentences"] for r in rs])
        med_w = np.median([r["word_count"] for r in rs])
        print(f"  {b}: median {med_s:.0f} sentences, {med_w:.0f} words")

# %% Cell 6 — Multi-question: cued vs uncued trace lengths across all 41 questions
print("\n" + "=" * 60)
print("MULTI-QUESTION: CUED vs UNCUED TRACE LENGTHS")
print("=" * 60)

with open(REPO_ROOT / "prompts" / "faith_metadata.json", "r", encoding="utf-8") as f:
    faith_meta = json.load(f)

# Discover all pn values that have both cued and uncued rollouts
pn_values = set()
for key, meta in faith_meta.items():
    if key.startswith("faith_cued_pn"):
        pn_values.add(meta["pn"])

multi_records = []  # (pn, condition, n_sentences, word_count)
for pn in sorted(pn_values):
    for condition in ["uncued", "cued"]:
        rd = REPO_ROOT / "prompts" / f"faith_{condition}_pn{pn}" / "qwen3-8b" / "rollouts"
        if not rd.exists():
            continue
        for rpath in rd.glob("*.json"):
            with open(rpath, "r", encoding="utf-8") as f:
                resp = json.load(f)
            sents = resp.get("sentences", [])
            cot_text = resp.get("cot_content", "")
            wc = len(cot_text.split()) if cot_text else 0
            multi_records.append((pn, condition, len(sents), wc))

print(f"Loaded {len(multi_records)} rollouts across {len(pn_values)} questions\n")

# Per-question comparison
print(f"{'pn':>5s}  {'n_unc':>5s} {'med_unc':>8s}  {'n_cue':>5s} {'med_cue':>8s}  {'diff%':>7s} {'U-p':>10s}")
print("-" * 62)

pn_diffs = []  # (pn, uncued_median, cued_median, p_value)
for pn in sorted(pn_values):
    unc_words = [wc for p, c, ns, wc in multi_records if p == pn and c == "uncued"]
    cue_words = [wc for p, c, ns, wc in multi_records if p == pn and c == "cued"]
    if not unc_words or not cue_words:
        continue
    med_u = np.median(unc_words)
    med_c = np.median(cue_words)
    diff_pct = 100.0 * (med_c - med_u) / med_u if med_u > 0 else 0.0
    _, p_val = mannwhitneyu(unc_words, cue_words, alternative="two-sided")
    pn_diffs.append((pn, med_u, med_c, p_val))
    sig = "*" if p_val < 0.05 else ""
    print(f"{pn:5d}  {len(unc_words):5d} {med_u:8.0f}  {len(cue_words):5d} {med_c:8.0f}  {diff_pct:+6.1f}% {p_val:10.4g} {sig}")

# Aggregate
n_shorter = sum(1 for _, mu, mc, _ in pn_diffs if mc < mu)
n_sig = sum(1 for _, _, _, p in pn_diffs if p < 0.05)
n_sig_shorter = sum(1 for _, mu, mc, p in pn_diffs if mc < mu and p < 0.05)
print(f"\nCued shorter than uncued: {n_shorter}/{len(pn_diffs)} questions")
print(f"Significant (p<0.05): {n_sig}/{len(pn_diffs)} questions")
print(f"Significant AND shorter: {n_sig_shorter}/{len(pn_diffs)} questions")

# Aggregate across all questions
all_unc = [wc for _, c, _, wc in multi_records if c == "uncued"]
all_cue = [wc for _, c, _, wc in multi_records if c == "cued"]
u_all, p_all = mannwhitneyu(all_unc, all_cue, alternative="two-sided")
print(f"\nPooled: uncued median={np.median(all_unc):.0f}, cued median={np.median(all_cue):.0f}, "
      f"U={u_all:.0f}, p={p_all:.4g}")

# --- Box plot: pooled across all questions ---
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(
    [all_unc, all_cue],
    tick_labels=[f"uncued\n(n={len(all_unc)})", f"cued\n(n={len(all_cue)})"],
    patch_artist=True, widths=0.5,
)
bp["boxes"][0].set_facecolor("tab:blue")
bp["boxes"][1].set_facecolor("tab:orange")
for patch in bp["boxes"]:
    patch.set_alpha(0.6)
ax.set_ylabel("Word count (~ tokens)")
ax.set_title(f"CoT Length: Cued vs Uncued — All {len(pn_diffs)} Questions Pooled")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "03_pooled_cued_vs_uncued.png", dpi=150)
print(f"Saved: {PLOTS_DIR / '03_pooled_cued_vs_uncued.png'}")
plt.close(fig)

# %% Cell 7 — Length diff vs hint adoption rate (gap)
print("\n" + "=" * 60)
print("LENGTH DIFF vs HINT ADOPTION GAP")
print("=" * 60)
print("Hypothesis: questions where the LLM uses the hint more have shorter cued CoTs\n")

from scipy.stats import pearsonr, spearmanr

# For each question, compute:
#   gap = P(cue_answer | cued) - P(cue_answer | uncued)
#   mean_length_diff = mean(cued_words) - mean(uncued_words)

scatter_data = []  # (pn, gap, mean_length_diff_pct, baseline_acc)
for pn in sorted(pn_values):
    meta_key = f"faith_cued_pn{pn}"
    if meta_key not in faith_meta:
        continue
    cue_answer = faith_meta[meta_key]["cue_answer"]
    gt_answer = faith_meta[meta_key]["gt_answer"]

    # Load answers from rollouts
    unc_answers = []
    unc_words = []
    for rpath in (REPO_ROOT / "prompts" / f"faith_uncued_pn{pn}" / "qwen3-8b" / "rollouts").glob("*.json"):
        with open(rpath) as f:
            resp = json.load(f)
        unc_answers.append(resp.get("processed_response_content", ""))
        cot = resp.get("cot_content", "")
        unc_words.append(len(cot.split()) if cot else 0)

    cue_answers = []
    cue_words = []
    for rpath in (REPO_ROOT / "prompts" / f"faith_cued_pn{pn}" / "qwen3-8b" / "rollouts").glob("*.json"):
        with open(rpath) as f:
            resp = json.load(f)
        cue_answers.append(resp.get("processed_response_content", ""))
        cot = resp.get("cot_content", "")
        cue_words.append(len(cot.split()) if cot else 0)

    if not unc_words or not cue_words:
        continue

    baseline_acc = sum(1 for a in unc_answers if a == gt_answer) / len(unc_answers)
    cued_acc = sum(1 for a in cue_answers if a == gt_answer) / len(cue_answers)
    gap = cued_acc - baseline_acc  # negative = hint hurts accuracy

    mean_diff_pct = 100.0 * (np.mean(cue_words) - np.mean(unc_words)) / np.mean(unc_words)

    scatter_data.append((pn, gap, mean_diff_pct, baseline_acc))

gaps = [g for _, g, _, _ in scatter_data]
diffs = [d for _, _, d, _ in scatter_data]
accs = [a for _, _, _, a in scatter_data]

r_pear, p_pear = pearsonr(gaps, diffs)
r_spear, p_spear = spearmanr(gaps, diffs)

print(f"{'pn':>5s}  {'gap':>6s}  {'len_diff%':>10s}  {'base_acc':>8s}")
print("-" * 38)
for pn, gap, diff, acc in scatter_data:
    print(f"{pn:5d}  {gap:+.3f}  {diff:+9.1f}%  {acc:7.1%}")

print(f"\nPearson  r={r_pear:.3f}, p={p_pear:.4g}")
print(f"Spearman r={r_spear:.3f}, p={p_spear:.4g}")

# --- Scatter plot colored by baseline accuracy ---
fig, ax = plt.subplots(figsize=(10, 7))
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = plt.cm.RdYlGn  # red=low accuracy, green=high accuracy

sc = ax.scatter(gaps, diffs, c=accs, cmap=cmap, norm=norm,
                s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Baseline accuracy P(correct | uncued)")

# Label each point with pn
for pn, gap, diff, _ in scatter_data:
    ax.annotate(str(pn), (gap, diff), fontsize=7, alpha=0.7,
                xytext=(4, 4), textcoords="offset points")

# Trend line
z = np.polyfit(gaps, diffs, 1)
x_line = np.linspace(min(gaps), max(gaps), 100)
ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6,
        label=f"OLS: slope={z[0]:.1f}, r={r_pear:.2f} (p={p_pear:.3g})")

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Correctness gap: P(correct|cued) - P(correct|uncued)")
ax.set_ylabel("Mean CoT length change (%): (cued - uncued) / uncued")
ax.set_title("Does using the hint shorten reasoning?  (color = baseline accuracy)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "04_length_vs_correctness_gap.png", dpi=150)
print(f"Saved: {PLOTS_DIR / '04_length_vs_correctness_gap.png'}")
plt.close(fig)

# %% Cell 8 — Hypothesis test: hint adoption vs CoT length
# If the model adopts the hint → shorter CoT (just goes with it)
# If the model resists the hint → longer CoT (has to reason past the cue)
print("\n" + "=" * 60)
print("HYPOTHESIS: HINT ADOPTION vs COT LENGTH")
print("=" * 60)

# Collect per-rollout data: (pn, group, word_count)
#   group: "uncued", "cued-adopted" (gave cue_answer), "cued-resisted" (gave gt_answer), "cued-other"
per_rollout = []

for pn in sorted(pn_values):
    meta_key = f"faith_cued_pn{pn}"
    if meta_key not in faith_meta:
        continue
    cue_answer = faith_meta[meta_key]["cue_answer"]
    gt_answer = faith_meta[meta_key]["gt_answer"]

    # Uncued rollouts
    unc_dir = REPO_ROOT / "prompts" / f"faith_uncued_pn{pn}" / "qwen3-8b" / "rollouts"
    if unc_dir.exists():
        for rpath in unc_dir.glob("*.json"):
            with open(rpath) as f:
                resp = json.load(f)
            cot = resp.get("cot_content", "")
            wc = len(cot.split()) if cot else 0
            per_rollout.append((pn, "uncued", wc))

    # Cued rollouts — classify by answer
    cue_dir = REPO_ROOT / "prompts" / f"faith_cued_pn{pn}" / "qwen3-8b" / "rollouts"
    if cue_dir.exists():
        for rpath in cue_dir.glob("*.json"):
            with open(rpath) as f:
                resp = json.load(f)
            ans = resp.get("processed_response_content", "")
            cot = resp.get("cot_content", "")
            wc = len(cot.split()) if cot else 0

            if ans == cue_answer:
                per_rollout.append((pn, "cued-adopted", wc))
            elif ans == gt_answer:
                per_rollout.append((pn, "cued-resisted", wc))
            else:
                per_rollout.append((pn, "cued-other", wc))

# Group word counts
from collections import defaultdict as _dd
groups = _dd(list)
for pn, grp, wc in per_rollout:
    groups[grp].append(wc)

GROUP_ORDER = ["uncued", "cued-resisted", "cued-adopted", "cued-other"]
GROUP_COLORS = {
    "uncued": "tab:blue",
    "cued-resisted": "tab:green",
    "cued-adopted": "tab:red",
    "cued-other": "tab:gray",
}
GROUP_LABELS = {
    "uncued": "Uncued\n(no hint)",
    "cued-resisted": "Cued → correct\n(resisted hint)",
    "cued-adopted": "Cued → cue answer\n(adopted hint)",
    "cued-other": "Cued → other\n(neither)",
}

print(f"\n{'Group':<20s} {'n':>6s} {'mean':>8s} {'median':>8s}")
print("-" * 46)
for g in GROUP_ORDER:
    vals = groups.get(g, [])
    if vals:
        print(f"{g:<20s} {len(vals):6d} {np.mean(vals):8.1f} {np.median(vals):8.1f}")

# Mann-Whitney: adopted vs resisted
if groups["cued-adopted"] and groups["cued-resisted"]:
    u, p = mannwhitneyu(groups["cued-adopted"], groups["cued-resisted"], alternative="two-sided")
    print(f"\nMann-Whitney adopted vs resisted: U={u:.0f}, p={p:.4g}")
if groups["cued-adopted"] and groups["uncued"]:
    u, p = mannwhitneyu(groups["cued-adopted"], groups["uncued"], alternative="two-sided")
    print(f"Mann-Whitney adopted vs uncued:   U={u:.0f}, p={p:.4g}")
if groups["cued-resisted"] and groups["uncued"]:
    u, p = mannwhitneyu(groups["cued-resisted"], groups["uncued"], alternative="two-sided")
    print(f"Mann-Whitney resisted vs uncued:   U={u:.0f}, p={p:.4g}")

# --- Box plot ---
fig, ax = plt.subplots(figsize=(10, 6))
plot_groups = [g for g in GROUP_ORDER if groups.get(g)]
box_data = [groups[g] for g in plot_groups]
box_labels = [f"{GROUP_LABELS[g]}\n(n={len(groups[g])})" for g in plot_groups]
box_colors = [GROUP_COLORS[g] for g in plot_groups]

bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5,
                showfliers=False)  # hide outliers for clarity
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay individual points (jittered)
for i, (g, vals) in enumerate(zip(plot_groups, box_data)):
    x = np.random.normal(i + 1, 0.06, size=len(vals))
    ax.scatter(x, vals, alpha=0.15, s=8, color=box_colors[i], zorder=3)

# Annotate medians
for i, vals in enumerate(box_data):
    med = np.median(vals)
    ax.text(i + 1, med, f" {med:.0f}", va="center", ha="left", fontsize=9,
            fontweight="bold", color="black")

ax.set_ylabel("CoT word count")
ax.set_title("CoT Length by Hint Adoption — All 41 Questions Pooled\n"
             "(Does adopting the hint shorten reasoning?)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "05_cot_length_by_hint_adoption.png", dpi=150)
print(f"\nSaved: {PLOTS_DIR / '05_cot_length_by_hint_adoption.png'}")
plt.close(fig)

# %% Cell 9 — Per-question: normalized length change by answer type
# For each question, compute mean CoT length for adopted vs resisted vs uncued
# Then show the *relative* change from uncued baseline
print("\n" + "=" * 60)
print("PER-QUESTION: NORMALIZED COT LENGTH BY ANSWER TYPE")
print("=" * 60)

per_q = defaultdict(lambda: defaultdict(list))  # pn -> group -> [word_counts]
for pn, grp, wc in per_rollout:
    per_q[pn][grp].append(wc)

# Compute baseline accuracy per question: P(correct | uncued)
q_baseline_acc = {}
for pn in sorted(per_q.keys()):
    meta_key = f"faith_cued_pn{pn}"
    if meta_key not in faith_meta:
        continue
    gt_answer = faith_meta[meta_key]["gt_answer"]
    unc_dir = REPO_ROOT / "prompts" / f"faith_uncued_pn{pn}" / "qwen3-8b" / "rollouts"
    if not unc_dir.exists():
        continue
    answers = []
    for rpath in unc_dir.glob("*.json"):
        with open(rpath) as f:
            resp = json.load(f)
        answers.append(resp.get("processed_response_content", ""))
    if answers:
        q_baseline_acc[pn] = sum(1 for a in answers if a == gt_answer) / len(answers)

q_data = []  # (pn, adopted_pct_change, resisted_pct_change, n_adopted, n_resisted, baseline_acc)
for pn in sorted(per_q.keys()):
    unc = per_q[pn].get("uncued", [])
    adopted = per_q[pn].get("cued-adopted", [])
    resisted = per_q[pn].get("cued-resisted", [])
    if not unc or np.mean(unc) == 0:
        continue
    baseline = np.mean(unc)
    adopted_pct = 100.0 * (np.mean(adopted) - baseline) / baseline if adopted else None
    resisted_pct = 100.0 * (np.mean(resisted) - baseline) / baseline if resisted else None
    bacc = q_baseline_acc.get(pn, 0.0)
    q_data.append((pn, adopted_pct, resisted_pct, len(adopted), len(resisted), bacc))

# --- Paired scatter: x = baseline accuracy, y = CoT length change ---
fig, ax = plt.subplots(figsize=(10, 7))

adopted_pts = [(bacc, a, na, pn) for pn, a, r, na, nr, bacc in q_data if a is not None]
resisted_pts = [(bacc, r, nr, pn) for pn, a, r, na, nr, bacc in q_data if r is not None]

if adopted_pts:
    xs_a, vals_a, ns_a, pns_a = zip(*adopted_pts)
    ax.scatter(xs_a, vals_a, c="tab:red", s=[max(20, n * 4) for n in ns_a],
               alpha=0.7, edgecolors="black", linewidth=0.5, label="Cued → cue answer (adopted)", zorder=3)
    for x, v, n, pn in adopted_pts:
        ax.annotate(str(pn), (x, v), fontsize=6, alpha=0.5, xytext=(4, 4), textcoords="offset points")

if resisted_pts:
    xs_r, vals_r, ns_r, pns_r = zip(*resisted_pts)
    ax.scatter(xs_r, vals_r, c="tab:green", s=[max(20, n * 4) for n in ns_r],
               alpha=0.7, edgecolors="black", linewidth=0.5, label="Cued → correct (resisted)", zorder=3)
    for x, v, n, pn in resisted_pts:
        ax.annotate(str(pn), (x, v), fontsize=6, alpha=0.5, xytext=(4, -8), textcoords="offset points")

# Connect adopted and resisted for same question
for pn, a, r, na, nr, bacc in q_data:
    if a is not None and r is not None:
        ax.plot([bacc, bacc], [a, r], color="gray", linewidth=0.5, alpha=0.4, zorder=1)

# Trend lines
if adopted_pts:
    xs_a_arr, vals_a_arr = np.array(xs_a), np.array(vals_a)
    z_a = np.polyfit(xs_a_arr, vals_a_arr, 1)
    x_line = np.linspace(0, 1, 100)
    r_a, p_a = pearsonr(xs_a_arr, vals_a_arr)
    ax.plot(x_line, np.polyval(z_a, x_line), "r--", alpha=0.5,
            label=f"Adopted trend: r={r_a:.2f} (p={p_a:.3g})")

if resisted_pts:
    xs_r_arr, vals_r_arr = np.array(xs_r), np.array(vals_r)
    z_r = np.polyfit(xs_r_arr, vals_r_arr, 1)
    r_r, p_r = pearsonr(xs_r_arr, vals_r_arr)
    ax.plot(x_line, np.polyval(z_r, x_line), "g--", alpha=0.5,
            label=f"Resisted trend: r={r_r:.2f} (p={p_r:.3g})")

ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
ax.set_xlabel("Baseline accuracy: P(correct | uncued)")
ax.set_ylabel("CoT length change from uncued baseline (%)")
ax.set_title("Per-Question: CoT Length Change by Answer Type vs Baseline Difficulty\n"
             "(Red below green = adopting hint shortens reasoning)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "06_per_question_length_by_answer_type.png", dpi=150)
print(f"Saved: {PLOTS_DIR / '06_per_question_length_by_answer_type.png'}")
plt.close(fig)

# Count how often adopted < resisted
n_both = sum(1 for _, a, r, _, _, _ in q_data if a is not None and r is not None)
n_adopted_shorter = sum(1 for _, a, r, _, _, _ in q_data if a is not None and r is not None and a < r)
print(f"\nQuestions with both adopted & resisted rollouts: {n_both}")
print(f"Adopted CoT shorter than resisted: {n_adopted_shorter}/{n_both} "
      f"({100*n_adopted_shorter/n_both:.0f}%)" if n_both > 0 else "")

# Sign test
if n_both > 0:
    from scipy.stats import binomtest
    result = binomtest(n_adopted_shorter, n_both, 0.5, alternative="greater")
    print(f"Sign test (adopted < resisted): p={result.pvalue:.4g}")

print("\nDone.")
