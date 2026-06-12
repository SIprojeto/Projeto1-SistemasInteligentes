"""
Projeto de Sistemas Inteligentes
Fase 4 — Comparação Final: ML vs Deep Learning
Gera tabelas e gráficos comparativos para o relatório
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os, json
from spark_display import imprimir_pandas, secao
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "fase4_comparacao")  # NOVO: pasta dedicada Fase 4
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WRITE_REPORTS = os.environ.get("WRITE_REPORTS", "1").lower() not in ("0", "false", "no")

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#795548"]
TYPE_COLORS = {
    "Spark ML": PALETTE[0],
    "Sklearn": PALETTE[2],
    "Deep Learning": PALETTE[3],
}
NN_MODELOS = ("MLP", "Deep MLP")
NN_COLORS = {"MLP": PALETTE[2], "Deep MLP": PALETTE[3]}


def _somente_redes_neuronais(df):
    if df.empty:
        return df
    return df[df["Modelo"].isin(NN_MODELOS)].copy()


def _grafico_comparacao_classificacao(df, filename, titulo_superior):
    """Barras Accuracy/F1 e AUC — equivalente visual aos CSV de comparação."""
    if df.empty:
        return
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo_superior, fontsize=13, fontweight="bold")
    x = np.arange(len(df))
    w = 0.35
    axes[0].bar(x - w / 2, df["Accuracy"], w, label="Accuracy", color=PALETTE[0], alpha=0.85)
    axes[0].bar(x + w / 2, df["F1-score"], w, label="F1-score", color=PALETTE[1], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Modelo"], rotation=22, ha="right", fontsize=9)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy & F1-score")
    axes[0].legend()
    score_min = float(df[["Accuracy", "F1-score"]].min().min())
    score_max = float(df[["Accuracy", "F1-score"]].max().max())
    axes[0].set_ylim(max(0, score_min - 0.05), min(1.02, score_max + 0.05))
    auc_min = float(df["AUC"].min())

    colors = [TYPE_COLORS.get(t, PALETTE[6]) for t in df.get("Tipo", ["?"] * len(df))]
    axes[1].bar(df["Modelo"], df["AUC"], color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC (↑ melhor discriminação)")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].set_ylim(max(0.45, auc_min - 0.05), 1.02)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_comparacao_regressao(df, filename, titulo_superior):
    """Barras RMSE/MAE e R² — regressão tip_amount."""
    if df.empty:
        return
    for col in ["RMSE", "MAE", "R²"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(titulo_superior, fontsize=13, fontweight="bold")
    x = np.arange(len(df))
    w = 0.35
    axes[0].bar(x - w / 2, df["RMSE"], w, label="RMSE", color=PALETTE[1], alpha=0.85)
    axes[0].bar(x + w / 2, df["MAE"], w, label="MAE", color=PALETTE[4], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Modelo"], rotation=22, ha="right", fontsize=9)
    axes[0].set_ylabel("Erro ($)")
    axes[0].set_title("RMSE & MAE (↓ melhor)")
    axes[0].legend()
    err_max = float(df[["RMSE", "MAE"]].max().max())
    axes[0].set_ylim(0, err_max * 1.12)

    colors = [TYPE_COLORS.get(t, PALETTE[6]) for t in df.get("Tipo", ["?"] * len(df))]
    axes[1].bar(df["Modelo"], df["R²"], color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² (↑ melhor)")
    axes[1].tick_params(axis="x", rotation=22)
    r2_min = float(df["R²"].min())
    r2_max = float(df["R²"].max())
    axes[1].set_ylim(max(0, r2_min - 0.05), min(1.02, r2_max + 0.05))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_radar_classificacao(df, filename, titulo):
    """Radar chart — só faz sentido para classificação (Accuracy, F1, AUC…)."""
    if df.empty:
        return
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    metrics_radar = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    angles = [n / float(len(metrics_radar)) * 2 * np.pi for n in range(len(metrics_radar))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=1.02)
    for i, row in df.iterrows():
        values = [row[m] for m in metrics_radar] + [row[metrics_radar[0]]]
        ax.plot(
            angles, values, "o", linewidth=1.8,
            linestyle=RADAR_STYLES[i % len(RADAR_STYLES)],
            color=PALETTE[i % len(PALETTE)], label=row["Modelo"], markersize=5,
        )
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=11)
    radar_min = float(df[metrics_radar].min().min())
    radar_max = float(df[metrics_radar].max().max())
    pad = max(0.02, (radar_max - radar_min) * 0.08)
    ax.set_ylim(max(0, radar_min - pad), min(1.02, radar_max + pad))
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12), fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_heatmap_classificacao(df, filename, titulo):
    if df.empty:
        return
    cols = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    heatmap_data = df.set_index("Modelo")[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    hm_min = float(heatmap_data.min().min())
    hm_max = float(heatmap_data.max().max())
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, linewidths=0.5,
        vmin=max(0, hm_min - 0.02), vmax=min(1.02, hm_max + 0.02),
        cbar_kws={"label": "Score (↑ melhor)"},
    )
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_heatmap_regressao(df, filename, titulo):
    if df.empty:
        return
    heatmap_data = df.set_index("Modelo")[["R²", "RMSE", "MAE"]].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax, linewidths=0.5,
        cbar_kws={"label": "R²↑ | RMSE/MAE↓"},
    )
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_media_grupo_classificacao(df, filename, titulo):
    if df.empty:
        return
    grouped = df.groupby("Tipo")[["Accuracy", "Precision", "Recall", "F1-score", "AUC"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.T.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[2], PALETTE[3]], alpha=0.85, edgecolor="white")
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC"], rotation=0)
    grp_min = float(grouped.min().min())
    grp_max = float(grouped.max().max())
    ax.set_ylim(max(0, grp_min - 0.05), min(1.02, grp_max + 0.05))
    ax.legend(title="Tipo")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_media_grupo_regressao(df, filename, titulo):
    if df.empty:
        return
    grouped = df.groupby("Tipo")[["RMSE", "MAE", "R²"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped.T.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[2], PALETTE[3]], alpha=0.85, edgecolor="white")
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("Métrica")
    ax.set_xticklabels(["RMSE (↓)", "MAE (↓)", "R² (↑)"], rotation=0)
    ax.legend(title="Tipo")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_rn_classificacao_direta(df, filename, titulo):
    """Comparação head-to-head MLP (sklearn) vs Deep MLP (Keras)."""
    df = _somente_redes_neuronais(df)
    if len(df) < 2:
        print(f"   ⚠ {filename} — dados insuficientes (correr fases 2 e 3)")
        return
    for col in ["Accuracy", "F1-score", "AUC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    metricas = [
        ("Accuracy", "Accuracy"),
        ("F1-score", "F1-score"),
        ("AUC", "AUC-ROC"),
    ]
    for ax, (col, label) in zip(axes, metricas):
        cores = [NN_COLORS.get(m, PALETTE[6]) for m in df["Modelo"]]
        ax.bar(df["Modelo"], df[col], color=cores, alpha=0.85, edgecolor="white")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(max(0, float(df[col].min()) - 0.05), min(1.02, float(df[col].max()) + 0.05))
        for i, v in enumerate(df[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_rn_regressao_direta(df, filename, titulo):
    df = _somente_redes_neuronais(df)
    if len(df) < 2:
        print(f"   ⚠ {filename} — dados insuficientes (correr fases 2 e 3)")
        return
    for col in ["RMSE", "MAE", "R²"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    specs = [("RMSE", "RMSE (↓)"), ("MAE", "MAE (↓)"), ("R²", "R² (↑)")]
    for ax, (col, label) in zip(axes, specs):
        cores = [NN_COLORS.get(m, PALETTE[6]) for m in df["Modelo"]]
        ax.bar(df["Modelo"], df[col], color=cores, alpha=0.85, edgecolor="white")
        ax.set_ylabel(label)
        ax.set_title(label)
        for i, v in enumerate(df[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   ✓ {filename}")


def _grafico_rn_resumo(clf_high, clf_given, reg):
    """Painel único: MLP vs Deep MLP nas 3 tarefas."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Redes Neuronais — MLP (sklearn) vs Deep MLP (TensorFlow/Keras)",
        fontsize=14, fontweight="bold",
    )
    paineis = [
        (clf_high, "F1-score", "tip_high\n(classificação)", 0),
        (clf_given, "F1-score", "tip_given\n(baseline)", 1),
        (reg, "R²", "tip_amount\n(regressão)", 2),
    ]
    for df_src, metrica, subtitulo, idx in paineis:
        ax = axes[idx]
        df = _somente_redes_neuronais(df_src)
        if df.empty or metrica not in df.columns:
            ax.set_title(subtitulo)
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax.transAxes)
            continue
        df[metrica] = pd.to_numeric(df[metrica], errors="coerce")
        cores = [NN_COLORS.get(m, PALETTE[6]) for m in df["Modelo"]]
        ax.bar(df["Modelo"], df[metrica], color=cores, alpha=0.85, edgecolor="white")
        ax.set_title(subtitulo, fontweight="bold")
        ax.set_ylabel(metrica)
        for i, v in enumerate(df[metrica]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].legend(
        handles=[
            mpatches.Patch(color=NN_COLORS["MLP"], label="MLP (sklearn)"),
            mpatches.Patch(color=NN_COLORS["Deep MLP"], label="Deep MLP (Keras)"),
        ],
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig_comparacao_redes_neuronais.png"), bbox_inches="tight", dpi=120)
    plt.close()
    print("   ✓ fig_comparacao_redes_neuronais.png")


RADAR_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]

print("=" * 60)
print("FASE 4 — COMPARAÇÃO FINAL ML vs DEEP LEARNING")
print("=" * 60)

# SI: síntese comparativa ML clássico vs Deep Learning (classificação + regressão)
# Spark: agrega resultados Spark ML (fase 2) com sklearn e TensorFlow para o relatório
# ─────────────────────────────────────────
# CARREGAR RESULTADOS REAIS DAS FASES 2 E 3
# ─────────────────────────────────────────
clf_ml_path = f"{OUTPUT_DIR}/resultados_classificacao.csv"
reg_ml_path = f"{OUTPUT_DIR}/resultados_regressao.csv"
sklearn_clf_path = f"{OUTPUT_DIR}/sklearn_classification_results.csv"
sklearn_reg_path = f"{OUTPUT_DIR}/sklearn_regression_results.csv"
unsup_path = f"{OUTPUT_DIR}/unsupervised_models_results.csv"
dl_path     = f"{OUTPUT_DIR}/resultados_deep_learning.json"

required_paths = [clf_ml_path, reg_ml_path, dl_path]
for path in required_paths:
    if not os.path.exists(path):
        print(f"\n[ERRO] Ficheiro não encontrado: {path}")
        print("       Corre primeiro as fases 2 e 3.")
        exit(1)

# Resultados ML (fase 2)
clf_ml = pd.read_csv(clf_ml_path)
reg_ml = pd.read_csv(reg_ml_path)
sklearn_clf = pd.read_csv(sklearn_clf_path) if os.path.exists(sklearn_clf_path) else pd.DataFrame()
sklearn_reg = pd.read_csv(sklearn_reg_path) if os.path.exists(sklearn_reg_path) else pd.DataFrame()
unsup_df = pd.read_csv(unsup_path) if os.path.exists(unsup_path) else pd.DataFrame()

# Resultados Deep Learning (fase 3)
with open(dl_path) as f:
    dl = json.load(f)

clf_ml["Tipo"] = "Spark ML"
if "AUC-ROC" in clf_ml.columns:
    clf_ml = clf_ml.rename(columns={"AUC-ROC": "AUC"})
if "PR-AUC" not in clf_ml.columns:
    clf_ml["PR-AUC"] = np.nan
if "Tipo" not in reg_ml.columns:
    reg_ml["Tipo"] = reg_ml["Modelo"].apply(
        lambda m: "Spark ML" if "Spark" in str(m) else "Sklearn"
    )

if not sklearn_clf.empty:
    if "AUC-ROC" in sklearn_clf.columns:
        sklearn_clf = sklearn_clf.rename(columns={"AUC-ROC": "AUC"})
    sklearn_clf["Tipo"] = "Sklearn"
    # Garantir todas as colunas esperadas
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "PR-AUC"]:
        if col not in sklearn_clf.columns:
            sklearn_clf[col] = np.nan

if not sklearn_reg.empty:
    sklearn_reg["Tipo"] = "Sklearn"
    if "R²" not in sklearn_reg.columns:
        sklearn_reg["R²"] = np.nan
    for col in ["RMSE", "MAE", "R²"]:
        if col not in sklearn_reg.columns:
            sklearn_reg[col] = np.nan

# SI: tabela única comparando Spark ML, sklearn e Deep MLP (fig15–fig19)
# Spark: resultados_classificacao.csv e resultados_regressao.csv gerados na fase 2 Spark
# Adicionar Deep MLP à tabela de classificação
dl_clf_row = pd.DataFrame([{
    "Modelo":    "Deep MLP",
    "Accuracy":  dl["deep_mlp_clf"]["accuracy"],
    "Precision": dl["deep_mlp_clf"]["precision"],
    "Recall":    dl["deep_mlp_clf"]["recall"],
    "F1-score":  dl["deep_mlp_clf"]["f1"],
    "AUC":       dl["deep_mlp_clf"]["auc"],
    "Tipo":      "Deep Learning",
}])

# Adicionar Deep MLP à tabela de regressão
dl_reg_row = pd.DataFrame([{
    "Modelo": "Deep MLP",
    "RMSE":   dl["deep_mlp_reg"]["rmse"],
    "MAE":    dl["deep_mlp_reg"]["mae"],
    "R²":     dl["deep_mlp_reg"]["r2"],
    "Tipo":   "Deep Learning",
}])

# Garantir colunas numéricas
for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "PR-AUC"]:
    if col in clf_ml.columns:
        clf_ml[col] = pd.to_numeric(clf_ml[col], errors="coerce")
    if not sklearn_clf.empty and col in sklearn_clf.columns:
        sklearn_clf[col] = pd.to_numeric(sklearn_clf[col], errors="coerce")
for col in ["RMSE", "MAE", "R²"]:
    if col in reg_ml.columns:
        reg_ml[col] = pd.to_numeric(reg_ml[col], errors="coerce")
    if not sklearn_reg.empty and col in sklearn_reg.columns:
        sklearn_reg[col] = pd.to_numeric(sklearn_reg[col], errors="coerce")

clf_given_df = pd.concat(
    [clf_ml[["Modelo","Tipo","Accuracy","Precision","Recall","F1-score","AUC","PR-AUC"]]] +
    ([sklearn_clf[["Modelo","Tipo","Accuracy","Precision","Recall","F1-score","AUC","PR-AUC"]]] if not sklearn_clf.empty else []) +
    [dl_clf_row.assign(**{"PR-AUC": np.nan})],
    ignore_index=True
)
reg_df = pd.concat(
    [reg_ml[["Modelo","Tipo","RMSE","MAE","R²"]]] +
    ([sklearn_reg[["Modelo","Tipo","RMSE","MAE","R²"]]] if not sklearn_reg.empty else []) +
    [dl_reg_row],
    ignore_index=True
)

# ── Classificação tip_high (tarefa principal SI) ────────────────
clf_high_path = f"{OUTPUT_DIR}/resultados_classificacao_tip_high.csv"
sklearn_high_path = f"{OUTPUT_DIR}/sklearn_classification_results_tip_high.csv"
clf_high_df = pd.DataFrame()
if os.path.exists(clf_high_path):
    clf_ml_high = pd.read_csv(clf_high_path)
    clf_ml_high["Tipo"] = "Spark ML"
    if "AUC-ROC" in clf_ml_high.columns:
        clf_ml_high = clf_ml_high.rename(columns={"AUC-ROC": "AUC"})
    sklearn_high = pd.read_csv(sklearn_high_path) if os.path.exists(sklearn_high_path) else pd.DataFrame()
    if not sklearn_high.empty:
        if "AUC-ROC" in sklearn_high.columns:
            sklearn_high = sklearn_high.rename(columns={"AUC-ROC": "AUC"})
        sklearn_high["Tipo"] = "Sklearn"
    dl_high_row = pd.DataFrame()
    if dl.get("deep_mlp_clf_tip_high"):
        h = dl["deep_mlp_clf_tip_high"]
        dl_high_row = pd.DataFrame([{
            "Modelo": "Deep MLP",
            "Accuracy": h["accuracy"],
            "Precision": h["precision"],
            "Recall": h["recall"],
            "F1-score": h["f1"],
            "AUC": h["auc"],
            "Tipo": "Deep Learning",
        }])
    parts_high = [clf_ml_high[["Modelo", "Tipo", "Accuracy", "Precision", "Recall", "F1-score", "AUC"]]]
    if not sklearn_high.empty:
        parts_high.append(sklearn_high[["Modelo", "Tipo", "Accuracy", "Precision", "Recall", "F1-score", "AUC"]])
    if not dl_high_row.empty:
        parts_high.append(dl_high_row)
    clf_high_df = pd.concat(parts_high, ignore_index=True)
    clf_high_df["PR-AUC"] = np.nan

# Classificação principal: tip_high (uniforme com regressão tip_amount)
if not clf_high_df.empty:
    clf_df = clf_high_df.copy()
else:
    print("\n[AVISO] resultados_classificacao_tip_high.csv não encontrado — fig15–fig19 usam tip_given.")
    clf_df = clf_given_df.copy()

if WRITE_REPORTS:
    clf_df.to_csv(f"{OUTPUT_DIR}/comparacao_classificacao.csv", index=False)
    reg_df.to_csv(f"{OUTPUT_DIR}/comparacao_regressao.csv", index=False)
    clf_given_df.to_csv(f"{OUTPUT_DIR}/comparacao_classificacao_tip_given.csv", index=False)
    if not clf_high_df.empty:
        clf_high_df.to_csv(f"{OUTPUT_DIR}/comparacao_classificacao_tip_high.csv", index=False)

clf_task = "tip_high (gorjeta alta ≥P75 vs baixa)" if not clf_high_df.empty else "tip_given (baseline)"
imprimir_pandas(clf_df, titulo=f"Comparação — Classificação {clf_task}")
imprimir_pandas(reg_df, titulo="Comparação — Regressão tip_amount")
imprimir_pandas(clf_given_df, titulo="Comparação — Classificação tip_given (baseline)")

print("\n── Figuras Fase 4 ──")

# 1) Comparação em barras — uma figura por tarefa
_grafico_comparacao_classificacao(
    clf_high_df.copy() if not clf_high_df.empty else clf_df.copy(),
    "fig_comparacao_tip_high.png",
    "Classificação tip_high — gorjeta alta (≥P75) vs baixa (tarefa principal SI)",
)
_grafico_comparacao_regressao(
    reg_df.copy(),
    "fig_comparacao_tip_amount.png",
    "Regressão tip_amount — previsão do valor da gorjeta ($)",
)
_grafico_comparacao_classificacao(
    clf_given_df.copy(),
    "fig_comparacao_tip_given.png",
    "Classificação tip_given — houve gorjeta? (baseline; classes desequilibradas)",
)

# 2) Radar — só classificação (tip_high + tip_given); regressão não usa radar
if not clf_high_df.empty:
    _grafico_radar_classificacao(
        clf_high_df.copy(),
        "fig15_radar_tip_high.png",
        "Radar — Classificação tip_high (gorjeta alta ≥P75 vs baixa)",
    )
_grafico_radar_classificacao(
    clf_given_df.copy(),
    "fig15_radar_tip_given.png",
    "Radar — Classificação tip_given (houve gorjeta? — baseline)",
)

# 3) Heatmaps — ficheiros separados por tarefa
if not clf_high_df.empty:
    _grafico_heatmap_classificacao(
        clf_high_df.copy(),
        "fig18_heatmap_tip_high.png",
        "Heatmap — Classificação tip_high",
    )
_grafico_heatmap_classificacao(
    clf_given_df.copy(),
    "fig18_heatmap_tip_given.png",
    "Heatmap — Classificação tip_given (baseline)",
)
_grafico_heatmap_regressao(
    reg_df.copy(),
    "fig18_heatmap_tip_amount.png",
    "Heatmap — Regressão tip_amount",
)

# 4) Síntese por grupo (Spark / Sklearn / DL) — fig17
_grafico_media_grupo_classificacao(
    clf_high_df.copy() if not clf_high_df.empty else clf_df.copy(),
    "fig17_classificacao_tip_high.png",
    "Classificação tip_high — Média por Grupo (Spark / Sklearn / DL)",
)
_grafico_media_grupo_classificacao(
    clf_given_df.copy(),
    "fig17_classificacao_tip_given.png",
    "Classificação tip_given — Média por Grupo (Spark / Sklearn / DL)",
)
_grafico_media_grupo_regressao(
    reg_df.copy(),
    "fig17_regressao_tip_amount.png",
    "Regressão tip_amount — Média por Grupo (Spark / Sklearn / DL)",
)

# 4b) Redes neuronais — MLP (sklearn) vs Deep MLP (TensorFlow/Keras)
print("\n── Redes Neuronais ──")
nn_rows = []
if not clf_high_df.empty:
    nn_h = _somente_redes_neuronais(clf_high_df.copy())
    nn_h["Tarefa"] = "tip_high"
    nn_rows.append(nn_h)
nn_g = _somente_redes_neuronais(clf_given_df.copy())
nn_g["Tarefa"] = "tip_given"
nn_rows.append(nn_g)
nn_r = _somente_redes_neuronais(reg_df.copy())
nn_r["Tarefa"] = "tip_amount"
nn_rows.append(nn_r)
nn_df = pd.concat(nn_rows, ignore_index=True)
if WRITE_REPORTS:
    nn_df.to_csv(f"{OUTPUT_DIR}/comparacao_redes_neuronais.csv", index=False)
imprimir_pandas(nn_df, titulo="Comparação — Redes Neuronais (MLP sklearn vs Deep MLP Keras)")

if not clf_high_df.empty:
    _grafico_rn_classificacao_direta(
        clf_high_df.copy(),
        "fig_comparacao_rn_tip_high.png",
        "Redes Neuronais — Classificação tip_high (MLP sklearn vs Deep MLP Keras)",
    )
_grafico_rn_classificacao_direta(
    clf_given_df.copy(),
    "fig_comparacao_rn_tip_given.png",
    "Redes Neuronais — Classificação tip_given (MLP sklearn vs Deep MLP Keras)",
)
_grafico_rn_regressao_direta(
    reg_df.copy(),
    "fig_comparacao_rn_tip_amount.png",
    "Redes Neuronais — Regressão tip_amount (MLP sklearn vs Deep MLP Keras)",
)
_grafico_rn_resumo(
    clf_high_df if not clf_high_df.empty else clf_df,
    clf_given_df,
    reg_df,
)

# 5) Supervisionado vs não supervisionado — fig19
if not unsup_df.empty:
    unsup_compare_df = unsup_df.copy()
else:
    unsup_compare_df = pd.DataFrame([{
        "Abordagem": "Autoencoder (DL, não supervisionado)",
        "Métrica": "Taxa de anomalia",
        "Valor": float(dl["autoencoder"]["anomaly_rate"]),
        "Interpretação": "Percentagem de amostras acima do limiar de reconstrução"
    }])
if WRITE_REPORTS:
    unsup_compare_df.to_csv(f"{OUTPUT_DIR}/comparacao_supervisionado_nao_supervisionado.csv", index=False)
imprimir_pandas(unsup_compare_df, titulo="Comparação Supervisionado vs Não Supervisionado")
best_clf_tmp = clf_df.loc[clf_df["F1-score"].idxmax()]
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ["Supervisionado (melhor F1)", "Não supervisionado (anomalia)"]
values = [float(best_clf_tmp["F1-score"]), float(dl["autoencoder"]["anomaly_rate"])]
bars = ax.bar(labels, values, color=[PALETTE[0], PALETTE[3]], alpha=0.85)
ax.set_ylabel("Score / Taxa")
ax.set_title("Síntese: Supervisionado vs Não Supervisionado", fontweight="bold")
ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig19_sup_vs_unsup.png", bbox_inches="tight", dpi=120)
plt.close()
print("   ✓ fig19_sup_vs_unsup.png")

print("\n" + "=" * 60)
print("SÍNTESE FINAL")
print("=" * 60)
best_clf = clf_df.loc[clf_df["F1-score"].idxmax()]
best_reg = reg_df.loc[reg_df["R²"].idxmax()]
print(f"\n  Melhor classificador ({clf_task}): {best_clf['Modelo']} (F1={best_clf['F1-score']:.4f}, AUC={best_clf['AUC']:.4f})")
print(f"  Melhor regressor     : {best_reg['Modelo']} (R²={best_reg['R²']:.4f}, RMSE={best_reg['RMSE']:.4f})")
ml_clf = clf_df[clf_df["Tipo"].isin(["Spark ML", "Sklearn"])]
dl_clf = clf_df[clf_df["Tipo"] == "Deep Learning"]
ml_reg = reg_df[reg_df["Tipo"].isin(["Spark ML", "Sklearn"])]
dl_reg = reg_df[reg_df["Tipo"] == "Deep Learning"]
print(f"\n  Média F1 ML           : {ml_clf['F1-score'].mean():.4f}")
print(f"  Média F1 Deep Learning: {dl_clf['F1-score'].mean():.4f}")
print(f"\n  Média R² ML           : {ml_reg['R²'].mean():.4f}")
print(f"  Média R² Deep Learning: {dl_reg['R²'].mean():.4f}")

print("\n✅ Fase 4 concluída! Figuras em outputs/fase4_comparacao/:")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith(".png"):
        print(f"   {f}")
