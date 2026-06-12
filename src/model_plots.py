"""
Figuras padronizadas por modelo — estrutura comparável entre algoritmos.

outputs/fase2_ml/ (ou fase3_dl_anomalias/)
  supervisionado/
    {model_slug}/
      classificacao_tip_given|tip_high/
        matriz_confusao.png
        curvas_roc_pr.png
        arvore_decisao.png   (opcional, só J48)
      regressao_tip_amount/
        previsto_vs_real.png
        residuos.png
  nao_supervisionado/
    {model_slug}/
      distribuicao_erro.png / espaco_latente.png   (autoencoder)
      clusters_pca.png                               (kmeans, dbscan)
"""

import os
import shutil
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    r2_score,
)
from sklearn.tree import plot_tree

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raízes de output por fase — sem pasta "modelos" intermédia
FASE2_ROOT = os.path.join(BASE_DIR, "outputs", "fase2_ml")
FASE3_ROOT = os.path.join(BASE_DIR, "outputs", "fase3_dl_anomalias")

# Slugs por fase
_FASE2_SLUGS = {
    "random_forest", "logistic_regression", "svm", "j48", "mlp_sklearn", "naive_bayes"
}
_FASE3_SLUGS = {
    "deep_mlp", "autoencoder", "kmeans", "dbscan"
}

# Mantido por retrocompatibilidade (não usado internamente)
MODELS_OUTPUT_ROOT = FASE2_ROOT


def _fase_root(model_slug: str) -> str:
    if model_slug in _FASE3_SLUGS:
        return FASE3_ROOT
    return FASE2_ROOT

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

PARADIGM_SUPERVISIONADO = "supervisionado"
PARADIGM_NAO_SUPERVISIONADO = "nao_supervisionado"

TASK_TIP_GIVEN = "classificacao_tip_given"
TASK_TIP_HIGH = "classificacao_tip_high"
TASK_REGRESSAO = "regressao_tip_amount"

SLUG_RANDOM_FOREST = "random_forest"
SLUG_LOGISTIC_REGRESSION = "logistic_regression"
SLUG_SVM = "svm"
SLUG_J48 = "j48"
SLUG_MLP_SKLEARN = "mlp_sklearn"
SLUG_NAIVE_BAYES = "naive_bayes"
SLUG_DEEP_MLP = "deep_mlp"
SLUG_AUTOENCODER = "autoencoder"
SLUG_KMEANS = "kmeans"
SLUG_DBSCAN = "dbscan"


def task_from_label_col(label_col: str) -> str:
    return TASK_TIP_HIGH if label_col == "tip_high" else TASK_TIP_GIVEN


def model_dir(
    model_slug: str,
    task: Optional[str] = None,
    paradigm: str = PARADIGM_SUPERVISIONADO,
) -> str:
    """
    Pasta de saída do modelo, organizada por fase.

    Fase 2 (modelos clássicos): outputs/fase2_ml/{slug}/{tarefa}/
    Fase 3 (DL + não-supervisionado): outputs/fase3_dl_anomalias/{slug}/   (nao-sup)
                                       outputs/fase3_dl_anomalias/{slug}/{tarefa}/  (sup)
    """
    root = _fase_root(model_slug)
    if paradigm == PARADIGM_NAO_SUPERVISIONADO:
        path = os.path.join(root, model_slug)
    else:
        if not task:
            raise ValueError("Tarefa obrigatória para modelos supervisionados")
        path = os.path.join(root, model_slug, task)
    os.makedirs(path, exist_ok=True)
    return path


def _rel_path(
    model_slug: str, filename: str, task: Optional[str] = None,
    paradigm: str = PARADIGM_SUPERVISIONADO,
) -> str:
    fase = "fase2_ml" if model_slug in _FASE2_SLUGS else "fase3_dl_anomalias"
    if paradigm == PARADIGM_NAO_SUPERVISIONADO:
        return f"{fase}/{model_slug}/{filename}"
    return f"{fase}/{model_slug}/{task}/{filename}"


def _class_labels(label_col: str):
    if label_col == "tip_high":
        return ["Gorjeta baixa", "Gorjeta alta"]
    return ["Sem gorjeta", "Com gorjeta"]


def salvar_matriz_confusao(
    y_true, y_pred, model_slug, task, titulo_modelo, label_col="tip_given",
):
    """Matriz de confusão padronizada."""
    cm = confusion_matrix(y_true, y_pred)
    classes = _class_labels(label_col)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )

    ax.set_ylabel("Classe real", fontsize=10)
    ax.set_xlabel("Classe prevista", fontsize=10)
    ax.set_title(f"Matriz de confusão — {titulo_modelo}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(model_dir(model_slug, task), "matriz_confusao.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'matriz_confusao.png', task)}")
    return cm


def salvar_curvas_roc_pr(
    y_true, y_proba, model_slug, task, titulo_modelo, label_col="tip_given",
):
    """Curvas ROC e Precision-Recall padronizadas."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Curvas ROC e PR — {titulo_modelo}", fontsize=12, fontweight="bold")

    axes[0].plot(fpr, tpr, color=PALETTE[0], linewidth=2, label=f"AUC-ROC = {auc_roc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    axes[0].set_xlabel("Taxa de falsos positivos", fontsize=10)
    axes[0].set_ylabel("Taxa de verdadeiros positivos", fontsize=10)
    axes[0].set_title("Curva ROC")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])

    axes[1].plot(recall, precision, color=PALETTE[2], linewidth=2, label=f"PR-AUC = {auc_pr:.4f}")
    axes[1].axhline(
        y=np.mean(y_true), color="red", linestyle="--", alpha=0.6,
        label=f"Baseline ({np.mean(y_true):.2f})",
    )
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("Curva Precision-Recall")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    plt.tight_layout()
    out = os.path.join(model_dir(model_slug, task), "curvas_roc_pr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'curvas_roc_pr.png', task)}")
    return auc_roc, auc_pr


def salvar_classificacao(
    y_true, y_pred, y_proba, model_slug, titulo_modelo, label_col="tip_given",
):
    """Gera o par de figuras obrigatório para todos os classificadores."""
    task = task_from_label_col(label_col)
    cm = salvar_matriz_confusao(y_true, y_pred, model_slug, task, titulo_modelo, label_col)
    salvar_curvas_roc_pr(y_true, y_proba, model_slug, task, titulo_modelo, label_col)
    return cm


def salvar_regressao(y_true, y_pred, model_slug, titulo_modelo, clip_nonneg=True):
    """Previsto vs real + resíduos — padrão para todos os regressores."""
    task = TASK_REGRESSAO
    out_dir = model_dir(model_slug, task)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if clip_nonneg:
        y_pred = np.clip(y_pred, 0, None)

    residuos = y_true - y_pred
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sample_n = min(3000, len(y_true))
    idx = np.random.default_rng(42).choice(len(y_true), sample_n, replace=False)
    ax.scatter(y_true[idx], y_pred[idx], alpha=0.25, s=10, color=PALETTE[2])
    lim_hi = max(1.0, float(np.percentile(y_true, 99)))
    lims = [0, min(20, lim_hi)]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Valor real ($)", fontsize=10)
    ax.set_ylabel("Valor previsto ($)", fontsize=10)
    ax.set_title(f"Previsto vs real — {titulo_modelo}\n(R² = {r2:.4f})", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "previsto_vs_real.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'previsto_vs_real.png', task)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Resíduos — {titulo_modelo}", fontsize=12, fontweight="bold")

    axes[0].scatter(y_pred[idx], residuos[idx], alpha=0.25, s=10, color=PALETTE[0])
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Valor previsto ($)", fontsize=10)
    axes[0].set_ylabel("Resíduo ($)", fontsize=10)
    axes[0].set_title("Resíduos vs previsto")

    axes[1].hist(residuos, bins=50, color=PALETTE[4], alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Resíduo ($)", fontsize=10)
    axes[1].set_ylabel("Frequência", fontsize=10)
    axes[1].set_title("Distribuição dos resíduos")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuos.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'residuos.png', task)}")


def salvar_arvore_decisao(
    clf, feature_names, model_slug, task, titulo_modelo,
    class_names=None, max_depth_plot=3,
):
    """Visualização da árvore J48 (extra, na mesma pasta da classificação)."""
    class_names = class_names or ["Sem gorjeta", "Com gorjeta"]
    out_dir = model_dir(model_slug, task)

    plt.figure(figsize=(22, 10))
    plot_tree(
        clf, max_depth=max_depth_plot,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True, fontsize=9,
    )
    plt.title(
        f"Árvore de decisão — {titulo_modelo} (profundidade {max_depth_plot} de {clf.get_depth()})",
        fontsize=13, fontweight="bold",
    )
    out = os.path.join(out_dir, "arvore_decisao.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'arvore_decisao.png', task)}")


def salvar_autoencoder_analise(
    history_ae, recon_errors, y_clf_te, threshold, latent, titulo_modelo="Autoencoder",
):
    """Figuras do autoencoder em nao_supervisionado/autoencoder/."""
    out_dir = model_dir(SLUG_AUTOENCODER, paradigm=PARADIGM_NAO_SUPERVISIONADO)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Erro de reconstrução — {titulo_modelo}", fontsize=12, fontweight="bold")

    axes[0].plot(history_ae.history["loss"], label="Treino", color=PALETTE[0])
    axes[0].plot(history_ae.history["val_loss"], label="Validação", color=PALETTE[1])
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].hist(
        recon_errors[y_clf_te == 1], bins=60, alpha=0.7,
        color=PALETTE[2], label="Com gorjeta", density=True,
    )
    axes[1].hist(
        recon_errors[y_clf_te == 0], bins=60, alpha=0.7,
        color=PALETTE[1], label="Sem gorjeta", density=True,
    )
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                    label=f"Limiar P95 = {threshold:.3f}")
    axes[1].set_title("Distribuição do erro de reconstrução")
    axes[1].set_xlabel("Erro")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribuicao_erro.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(SLUG_AUTOENCODER, 'distribuicao_erro.png', paradigm=PARADIGM_NAO_SUPERVISIONADO)}")

    sample_lat = np.random.default_rng(42).choice(
        len(latent), min(3000, len(latent)), replace=False,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        latent[sample_lat, 0], latent[sample_lat, 1],
        c=y_clf_te[sample_lat], cmap="coolwarm", alpha=0.4, s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Gorjeta (0/1)")
    ax.set_title(f"Espaço latente — {titulo_modelo}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Latente 1")
    ax.set_ylabel("Latente 2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "espaco_latente.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(SLUG_AUTOENCODER, 'espaco_latente.png', paradigm=PARADIGM_NAO_SUPERVISIONADO)}")


def salvar_clusters_pca(X, labels, model_slug, titulo_modelo, color_by=None, sample_n=3000):
    """Visualização 2D (PCA) dos clusters em nao_supervisionado/{slug}/."""
    from sklearn.decomposition import PCA

    out_dir = model_dir(model_slug, paradigm=PARADIGM_NAO_SUPERVISIONADO)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(sample_n, len(X)), replace=False)
    X_s = X[idx]
    labels_s = labels[idx]
    color_s = color_by[idx] if color_by is not None else None

    coords = PCA(n_components=2, random_state=42).fit_transform(X_s)

    fig, axes = plt.subplots(1, 2 if color_s is not None else 1, figsize=(12 if color_s is not None else 6, 5))
    if color_s is None:
        axes = [axes]
    else:
        axes = list(axes)

    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=labels_s, cmap="tab10", alpha=0.5, s=10)
    axes[0].set_title(f"Clusters — {titulo_modelo}")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(sc0, ax=axes[0], label="Cluster")

    if color_s is not None:
        sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=color_s, cmap="coolwarm", alpha=0.5, s=10)
        axes[1].set_title("Mesma projeção — gorjeta real")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        plt.colorbar(sc1, ax=axes[1], label="Gorjeta (0/1)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clusters_pca.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'clusters_pca.png', paradigm=PARADIGM_NAO_SUPERVISIONADO)}")


def salvar_ablation_estudo(ablation_results, model_slug=SLUG_DEEP_MLP, task=TASK_TIP_HIGH):
    """Ablation study — figura extra na pasta supervisionada do Deep MLP."""
    if not ablation_results:
        return
    out_dir = model_dir(model_slug, task)

    arq_labels = [r["Arquitectura"] for r in ablation_results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Ablation study — comparação de arquitecturas Deep MLP",
        fontsize=13, fontweight="bold",
    )

    axes[0].bar(arq_labels, [r["F1-score"] for r in ablation_results],
                color=PALETTE[0], alpha=0.85, edgecolor="white")
    axes[0].set_title("F1-score")
    axes[0].set_ylabel("F1-score")
    axes[0].set_ylim(0, 1)

    axes[1].bar(arq_labels, [r["AUC-ROC"] for r in ablation_results],
                color=PALETTE[2], alpha=0.85, edgecolor="white")
    axes[1].set_title("AUC-ROC")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_ylim(0, 1)

    axes[2].bar(arq_labels, [r["Epocas"] for r in ablation_results],
                color=PALETTE[3], alpha=0.85, edgecolor="white")
    axes[2].set_title("Épocas até convergência")
    axes[2].set_ylabel("Nº épocas")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_estudo.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   OK {_rel_path(model_slug, 'ablation_estudo.png', task)}")
