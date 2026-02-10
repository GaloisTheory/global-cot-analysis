# %%
import json
import matplotlib.pyplot as plt
import numpy as np

with open("unfaithfulness_verification_all.json") as f:
    data = json.load(f)

qs = data["per_question"]

# %%
pns = [q["pn"] for q in qs]
gaps = [q["cue_response_gap"] for q in qs]
uf_rates = [q["unfaithfulness"]["unfaithfulness_rate"] for q in qs]

# Replace None with 0 for questions with no cue followers
uf_rates = [r if r is not None else 0 for r in uf_rates]

# %%
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(gaps, uf_rates, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)

# Label each point with pn
for pn, g, u in zip(pns, gaps, uf_rates):
    ax.annotate(str(pn), (g, u), fontsize=7, alpha=0.6,
                xytext=(4, 4), textcoords="offset points")

# Reference lines
ax.axvline(x=0.15, color="orange", linestyle="--", alpha=0.4, label="gap threshold (15%)")
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="50% unfaithful")

ax.set_xlabel("Cue Response Gap (p_cue_cued - p_cue_uncued)")
ax.set_ylabel("Unfaithfulness Rate (no keyword mention | chose cue answer)")
ax.set_title(f"Cue Response Gap vs Unfaithfulness Rate ({data['model']}, n={len(qs)})")
ax.legend(fontsize=9)
ax.set_xlim(-0.1, 1.05)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# %%
# P(originally correct) vs unfaithfulness rate
p_correct = [q["uncued"]["p_gt"] for q in qs]

fig, ax = plt.subplots(figsize=(9, 6))

sc = ax.scatter(p_correct, uf_rates, c=gaps, cmap="RdYlGn_r", s=60, alpha=0.8,
                edgecolors="white", linewidth=0.5, vmin=0, vmax=1)
cbar = fig.colorbar(sc, ax=ax, label="Cue Response Gap")

for pn, pc, u in zip(pns, p_correct, uf_rates):
    ax.annotate(str(pn), (pc, u), fontsize=7, alpha=0.6,
                xytext=(4, 4), textcoords="offset points")

ax.set_xlabel("P(correct | uncued)")
ax.set_ylabel("Unfaithfulness Rate (no keyword mention | chose cue answer)")
ax.set_title(f"Original Accuracy vs Unfaithfulness Rate ({data['model']}, n={len(qs)})")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# %%
# Original accuracy vs cue response gap, colored by unfaithfulness rate
fig, ax = plt.subplots(figsize=(9, 6))

sc = ax.scatter(p_correct, gaps, c=uf_rates, cmap="RdYlGn_r", s=60, alpha=0.8,
                edgecolors="white", linewidth=0.5, vmin=0, vmax=1)
fig.colorbar(sc, ax=ax, label="Unfaithfulness Rate")

for pn, pc, g in zip(pns, p_correct, gaps):
    ax.annotate(str(pn), (pc, g), fontsize=7, alpha=0.6,
                xytext=(4, 4), textcoords="offset points")

ax.axhline(y=0.15, color="orange", linestyle="--", alpha=0.4, label="gap threshold (15%)")
ax.legend(fontsize=9)

ax.set_xlabel("P(correct | uncued)")
ax.set_ylabel("Cue Response Gap (p_cue_cued - p_cue_uncued)")
ax.set_title(f"Original Accuracy vs Cue Response Gap ({data['model']}, n={len(qs)})")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.1, 1.05)

plt.tight_layout()
plt.show()

# %%
