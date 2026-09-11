import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from regression_utils import find_data_file, ols_with_inference, ols_beta_only, logit_with_inference, logit_beta_only, bc_ci_with_point

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")
BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

def to_float(s):
    return float(s.strip().replace(",", "."))

def de_num(x, decimals=3):
    """Formatiert eine Zahl im deutschen Stil (Komma statt Punkt, echtes
    Minuszeichen statt Bindestrich) fuer Beschriftungen in den Abbildungen,
    damit sie konsistent mit dem Fliesstext im Paper sind."""
    s = f"{x:.{decimals}f}"
    s = s.replace("-", "−")
    s = s.replace(".", ",")
    return s

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 150,
})
MAIN_COLOR = "#2C4A6E"     # gedecktes Blau
FIT_COLOR = "#8C2F2F"      # gedecktes Rot fuer Fit-Kurve
CONTROL_COLOR = "#666666"  # Grau fuer Kontrollbedingung

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {name}.png / {name}.pdf")


def wilson_ci(k, n, alpha=0.05):
    """Wilson-Score-Konfidenzintervall fuer einen Anteil (robuster als Normal-
    approximation bei kleinen n / Anteilen nahe 0 oder 1)."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return center - half, center + half


# ===================================================================================
# Abbildung 6: H1 - Preisexperiment (Annahmequote vs. Erstforderung)
# ===================================================================================
with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))
final = [r for r in rows if r["Codierung"].strip() != ""
         and r["Manipulationscheck_Preis"].strip() == "Ja"
         and r["Manipulationscheck_Validitaet"].strip() == "Ja"]

treat_rows = [r for r in final if r["Treatment"] != "849"]
treat = np.array([to_float(r["Treatment"]) for r in treat_rows])
y = np.array([to_float(r["Codierung"]) for r in treat_rows])
treat_k = treat / 1000.0
treat_k_c = treat_k - treat_k.mean()
X = np.column_stack([treat_k_c, treat_k_c ** 2])
res = logit_with_inference(X, y)
beta = res["beta"]

conds = sorted(set(int(r["Treatment"]) for r in final))
means, los, his, ns = [], [], [], []
for c in conds:
    grp = [int(r["Codierung"]) for r in final if int(r["Treatment"]) == c]
    k, n = sum(grp), len(grp)
    lo, hi = wilson_ci(k, n)
    means.append(k / n); los.append(lo); his.append(hi); ns.append(n)

# Kontrollbedingung (849 €) ist strukturell kein Punkt auf der "Erstforderung"-Skala
# der Treatments (kein zweistufiges DITF-Verfahren, keine Konzession, vgl. 3.1) und
# gehoert deshalb gar nicht auf die x-Achse. Sie wird stattdessen als horizontale
# Referenzlinie eingezeichnet (Standardweg, um einen Vergleichswert zu zeigen, ohne
# die x-Achse fuer einen einzelnen, kategorial andersartigen Punkt zu verbiegen).
fig, ax = plt.subplots(figsize=(6.3, 4.2))

treat_grid = np.linspace(1500, 10000, 200)
tg_k_c = (treat_grid / 1000.0) - treat_k.mean()
eta = beta[0] + beta[1] * tg_k_c + beta[2] * tg_k_c ** 2
p_fit = 1 / (1 + np.exp(-eta))
ax.plot(treat_grid, p_fit, color=FIT_COLOR, lw=2, label="Angepasste quadratische Logit-Kurve\n(nur Treatmentbedingungen)")

conds_arr = np.array(conds, dtype=float)
means_arr = np.array(means)
yerr = np.array([means_arr - np.array(los), np.array(his) - means_arr])
treatment_mask = conds_arr != 849
control_mean = means_arr[~treatment_mask][0]
control_n = np.array(ns)[~treatment_mask][0]

ax.axhline(control_mean, color=CONTROL_COLOR, lw=1.4, ls="--",
           label=f"Kontrollbedingung (849 €): {de_num(control_mean * 100, 1)}% (n={control_n})")
ax.errorbar(conds_arr[treatment_mask], means_arr[treatment_mask], yerr=yerr[:, treatment_mask],
            fmt="o", color=MAIN_COLOR, capsize=3, ms=6, label="Beobachtete Annahmequote je Bedingung\n(95%-Wilson-KI)")

his_arr = np.array(his)
for c, m, n, hi in zip(conds, means, ns, his_arr):
    if c == 849:
        continue
    ax.annotate(f"n={n}", (c, hi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="#555555")

ax.set_xlabel("Erstforderung (Euro)")
ax.set_ylabel("Annahmequote der Zielforderung")
ax.set_ylim(-0.02, 0.48)
ax.set_xlim(500, 11200)
ax.set_xticks([c for c in conds if c != 849])
ax.set_xticklabels([f"{c:,}".replace(",", ".") for c in conds if c != 849])
# Legende unter die Datenflaeche setzen statt in eine Ecke - bei den hohen
# Fehlerbalken (bis zu ~0.41) wuerde eine Ecken-Legende sonst mit Punkten/
# Beschriftungen kollidieren.
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, framealpha=0.9)
fig.tight_layout()
save(fig, "Abbildung_6_h1_preisexperiment")

# ===================================================================================
# Abbildung 7: H1 - Bewertungsexperiment (Punktzahl vs. Forderung)
# ===================================================================================
with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows2 = list(csv.DictReader(f, delimiter=";"))
final2 = [r for r in rows2 if r["Manipulationscheck_Rolle"].strip() == "Ja"
          and r["Manipulationscheck_Forderung"].strip() == "Ja"]

ford = np.array([to_float(r["Forderung_des_Studierenden"]) for r in final2])
punkte = np.array([to_float(r["Final_vergebene_Punktzahl"]) for r in final2])
ford_c = ford - ford.mean()
X2 = np.column_stack([ford_c, ford_c ** 2])
res2 = ols_with_inference(X2, punkte)
beta2 = res2["beta"]

conds2 = sorted(set(int(r["Forderung_des_Studierenden"]) for r in final2))
means2, ses2, ns2 = [], [], []
for c in conds2:
    grp = [to_float(r["Final_vergebene_Punktzahl"]) for r in final2 if int(r["Forderung_des_Studierenden"]) == c]
    means2.append(np.mean(grp)); ses2.append(np.std(grp, ddof=1) / np.sqrt(len(grp))); ns2.append(len(grp))

fig, ax = plt.subplots(figsize=(6.3, 4.2))
ford_grid = np.linspace(9, 15, 200)
fg_c = ford_grid - ford.mean()
y_fit = beta2[0] + beta2[1] * fg_c + beta2[2] * fg_c ** 2
ax.plot(ford_grid, y_fit, color=FIT_COLOR, lw=2, label="Angepasste quadratische OLS-Kurve")
ci95 = np.array(ses2) * 1.96
ax.errorbar(conds2, means2, yerr=ci95, fmt="o", color=MAIN_COLOR,
            capsize=3, ms=6, label="Mittelwert je Bedingung (± 95%-KI)")
# Label oberhalb des oberen KI-Endes platzieren, nicht ueber dem Punkt selbst -
# sonst laeuft der Fehlerbalken bei langen KIs mitten durch die Beschriftung.
for c, m, n, ci in zip(conds2, means2, ns2, ci95):
    ax.annotate(f"n={n}", (c, m + ci), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8, color="#555555")
ax.set_xlabel("Forderung des Studierenden (Punkte)")
ax.set_ylabel("Final vergebene Punktzahl")
ax.set_xticks(conds2)
ax.set_ylim(ax.get_ylim()[0], max(np.array(means2) + ci95) + 0.35)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
fig.tight_layout()
save(fig, "Abbildung_7_h1_bewertungsexperiment")

# ===================================================================================
# Abbildung 9: H2 - Indirekte Effekte im Preisexperiment (Forest Plot)
# ===================================================================================
# Wichtig: hier "treat_rows" (ohne Kontrollbedingung, N=195) verwenden, nicht
# "final" (N=246) - die Kontrollbedingung (849 EUR) ist kein Punkt auf der
# "Erstforderung"-Skala und darf nicht als X-Wert in die Mediationsanalyse
# eingehen, aus denselben Gruenden wie beim H1-Kurvenfit oben (vgl. 3.1).
X_all = np.array([to_float(r["Treatment"]) for r in treat_rows]) / 1000.0
X_all_c = X_all - X_all.mean()
M1 = np.array([to_float(r["Wahrnehmungskontrast"]) for r in treat_rows])
M2 = np.array([to_float(r["Konzession"]) for r in treat_rows])
M3 = np.array([to_float(r["Boomerang_Variable"]) for r in treat_rows])
Y_all = np.array([to_float(r["Codierung"]) for r in treat_rows])

a1 = ols_beta_only(X_all_c.reshape(-1, 1), M1)[1]
a2 = ols_beta_only(X_all_c.reshape(-1, 1), M2)[1]
a3 = ols_beta_only(X_all_c.reshape(-1, 1), M3)[1]
full_res = logit_with_inference(np.column_stack([X_all_c, M1, M2, M3]), Y_all)
b1, b2, b3 = full_res["beta"][2], full_res["beta"][3], full_res["beta"][4]
points = np.array([a1 * b1, a2 * b2, a3 * b3])

n = len(Y_all)
N_BOOT = 5000
boot = np.zeros((N_BOOT, 3))
for i in range(N_BOOT):
    idx = np.random.randint(0, n, n)
    Xb, Mb1, Mb2, Mb3, Yb = X_all_c[idx], M1[idx], M2[idx], M3[idx], Y_all[idx]
    if Yb.min() == Yb.max():
        boot[i, :] = np.nan
        continue
    aa1 = ols_beta_only(Xb.reshape(-1, 1), Mb1)[1]
    aa2 = ols_beta_only(Xb.reshape(-1, 1), Mb2)[1]
    aa3 = ols_beta_only(Xb.reshape(-1, 1), Mb3)[1]
    bf = logit_beta_only(np.column_stack([Xb, Mb1, Mb2, Mb3]), Yb)
    boot[i, 0] = aa1 * bf[2]
    boot[i, 1] = aa2 * bf[3]
    boot[i, 2] = aa3 * bf[4]

labels_h2 = ["Kontrasteffekt", "Konzession\n(Aufwärtsphase)", "Reaktanzindex\n(Abwärtsphase)"]
los_h2, his_h2 = [], []
for i in range(3):
    lo, hi = bc_ci_with_point(boot[:, i], points[i])
    los_h2.append(lo); his_h2.append(hi)

fig, ax = plt.subplots(figsize=(7.4, 3.2))
y_pos = np.arange(3)[::-1]
colors = [MAIN_COLOR if not (lo > 0 or hi < 0) else FIT_COLOR for lo, hi in zip(los_h2, his_h2)]
for yi, pt, lo, hi, col in zip(y_pos, points, los_h2, his_h2, colors):
    ax.plot([lo, hi], [yi, yi], color=col, lw=2)
    ax.plot(pt, yi, "o", color=col, ms=8)
    label = f"{de_num(pt)} [{de_num(lo)}, {de_num(hi)}]"
    ax.annotate(label, xy=(hi, yi), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8.5, color="#222222")
ax.axvline(0, color="#999999", lw=1, ls="--")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_h2)
ax.set_xlabel("Indirekter Effekt (a × b) mit 95%-BC-Bootstrap-KI")
ax.set_ylim(-0.7, 2.7)
ax.set_xlim(min(los_h2) - 0.02, max(his_h2) + 0.16)
fig.tight_layout()
save(fig, "Abbildung_8_h2_indirekte_effekte")

# ===================================================================================
# Abbildung 10: H3 - Mediationsmodell Bewertungsexperiment (Pfaddiagramm)
# ===================================================================================
M_h3 = np.array([to_float(r["Boomerang_Variable"]) for r in final2])
res_a3 = ols_with_inference(ford_c.reshape(-1, 1), M_h3)
a_h3 = res_a3["beta"][1]
X_full3 = np.column_stack([ford_c, M_h3])
res_full3 = ols_with_inference(X_full3, punkte)
b_h3 = res_full3["beta"][2]
c_prime_h3 = res_full3["beta"][1]
res_total3 = ols_with_inference(ford_c.reshape(-1, 1), punkte)
c_h3 = res_total3["beta"][1]

def p_str(p):
    if p < 0.001: return "p<.001"
    return f"p={p:.3f}"

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.4", fc="#EFEFEF", ec="#333333")
ax.text(1, 1, "Forderung\n(X)", ha="center", va="center", bbox=box_style, fontsize=10)
ax.text(5, 5, "Reaktanz /\nReaktanzindex\n(M)", ha="center", va="center", bbox=box_style, fontsize=10)
ax.text(9, 1, "Vergebene\nPunktzahl (Y)", ha="center", va="center", bbox=box_style, fontsize=10)

arrow_style = dict(arrowstyle="-|>", color="#333333", lw=1.6, mutation_scale=16)
ax.annotate("", xy=(4.1, 4.6), xytext=(1.7, 1.6), arrowprops=arrow_style)
label_bg = dict(facecolor="white", edgecolor="none", pad=1.5)
ax.text(2.6, 3.4, f"a = {de_num(a_h3)}***", fontsize=9, color=MAIN_COLOR, bbox=label_bg)
ax.annotate("", xy=(8.1, 1.6), xytext=(5.9, 4.6), arrowprops=arrow_style)
ax.text(6.9, 3.4, f"b = {de_num(b_h3)}**", fontsize=9, color=MAIN_COLOR, bbox=label_bg)
ax.annotate("", xy=(7.9, 1.0), xytext=(2.1, 1.0), arrowprops=arrow_style)
ax.text(5, 0.55, f"c' (direkt) = {de_num(c_prime_h3)}***  |  c (total) = {de_num(c_h3)}**",
        ha="center", fontsize=9, color=FIT_COLOR)

fig.tight_layout()
save(fig, "Abbildung_9_h3_mediationsmodell")

# ===================================================================================
# Abbildung 11: H4 - Fairness -> Reaktanz (Streudiagramm mit Regressionsgerade)
# ===================================================================================
fairness = np.array([to_float(r["Wahrgenommene_Fairness"]) for r in final2])
reaktanz = np.array([to_float(r["Boomerang_Variable"]) for r in final2])
fairness_c = fairness - fairness.mean()
res_h4 = ols_with_inference(fairness_c.reshape(-1, 1), reaktanz)
beta_h4 = res_h4["beta"]

rng = np.random.default_rng(7)
jitter_x = fairness + rng.uniform(-0.12, 0.12, size=len(fairness))
jitter_y = reaktanz + rng.uniform(-0.12, 0.12, size=len(reaktanz))

fig, ax = plt.subplots(figsize=(6.3, 4.2))
ax.scatter(jitter_x, jitter_y, s=14, color=MAIN_COLOR, alpha=0.45, edgecolors="none", label="Einzelne Fälle (leicht gejittert)")
fair_grid = np.linspace(fairness.min(), fairness.max(), 100)
y_line = beta_h4[0] + beta_h4[1] * (fair_grid - fairness.mean())
ax.plot(fair_grid, y_line, color=FIT_COLOR, lw=2.2, label=f"Regressionsgerade (β = {de_num(beta_h4[1])}, p<,001)")
ax.set_xlabel("Wahrgenommene Fairness der Erstforderung (1-5)")
ax.set_ylabel("Reaktanz (Reaktanzindex, 1-5)")
ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
fig.tight_layout()
save(fig, "Abbildung_10_h4_fairness_reaktanz")

print("\nAlle Abbildungen erstellt in:", OUT_DIR)
