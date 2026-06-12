"""
SVM — Support Vector Machine (sklearn).

Classificação e Regressão com kernel RBF.
Nota: treinado num subsample estratificado de 1500+1500 amostras por limitações
computacionais do SVM (complexidade O(n²) em memória).
Em produção, usaria LinearSVC que escala melhor.
"""

import os
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from model_plots import SLUG_SVM, salvar_classificacao, salvar_regressao

RANDOM_STATE = 42
SUBSAMPLE    = 3000        # Total de amostras (1500 classe 0 + 1500 classe 1)
N_PER_CLASS  = SUBSAMPLE // 2  # = 1500 por classe (estratificado 50/50)


# SI: SVM (kernel RBF) para classificação tip_given — requisito clássico do enunciado SI
# Spark: features vindas do Pipeline Spark; subsample estratificado por limite O(n²) do SVM em sklearn
def treinar_svm_clf(X_train, y_train, X_test, y_test,
                    figures_dir=None, label_col="tip_given", model_suffix=""):
    print("\n" + "="*60)
    print("SVM — Classificação (kernel RBF)")
    print("="*60)

    # ── BALANCEAMENTO ESTRATIFICADO 50/50 ──────────────────────────────────
    # CORREÇÃO: amostrar exactamente N_PER_CLASS de cada classe para corrigir
    # o desequilíbrio antes do treino. Conjunto de TESTE permanece desequilibrado.
    rng  = np.random.default_rng(RANDOM_STATE)
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    n0   = min(N_PER_CLASS, len(idx0))
    n1   = min(N_PER_CLASS, len(idx1))
    sampled_idx = np.concatenate([
        rng.choice(idx0, size=n0, replace=False),
        rng.choice(idx1, size=n1, replace=False),
    ])
    rng.shuffle(sampled_idx)
    X_tr, y_tr = X_train[sampled_idx], y_train[sampled_idx]

    print(f"   Subsample estratificado: {n0} cls0 + {n1} cls1 = {n0+n1} total")
    print(f"   (SVM com kernel RBF tem complexidade O(n²) — subsample necessário)")
    print(f"   Conjunto de TESTE: {len(y_test):,} amostras (proporção real, não balanceado)")
    print(f"   Alternativa produção: LinearSVC ou aproximação Nyström")
    print(f"   Parâmetros: kernel=rbf, C=1.0, gamma=scale")

    clf = SVC(kernel="rbf", C=1.0, gamma="scale",
              class_weight="balanced",
              probability=True, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    pred  = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo":    "SVM",
        "Accuracy":  f"{accuracy_score(y_test, pred):.4f}",
        "Precision": f"{precision_score(y_test, pred, zero_division=0):.4f}",
        "Recall":    f"{recall_score(y_test, pred, zero_division=0):.4f}",
        "F1-score":  f"{f1_score(y_test, pred, zero_division=0):.4f}",
        "AUC-ROC":   f"{roc_auc_score(y_test, proba):.4f}",
    }
    print(f"   Accuracy: {metrics['Accuracy']} | F1: {metrics['F1-score']} | AUC: {metrics['AUC-ROC']}")

    try:
        salvar_classificacao(y_test, pred, proba, SLUG_SVM, "SVM", label_col)
    except Exception as e:
        print(f"   ⚠ Não foi possível gerar figuras SVM: {e}")

    return metrics


# SI: SVR para regressão de tip_amount (comparação ML clássico vs DL na fase 4)
def treinar_svm_reg(X_train, y_train, X_test, y_test):
    print("\n── SVR — Regressão (kernel RBF) ──")

    if X_train.shape[0] > SUBSAMPLE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_train.shape[0], size=SUBSAMPLE, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
        print(f"   Subsample: {SUBSAMPLE} amostras (mesmo motivo do SVC)")
    else:
        X_tr, y_tr = X_train, y_train

    reg = SVR(kernel="rbf", C=1.0, gamma="scale")
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_test)

    metrics = {
        "Modelo": "SVM",
        "RMSE":   f"{np.sqrt(mean_squared_error(y_test, pred)):.4f}",
        "MAE":    f"{mean_absolute_error(y_test, pred):.4f}",
        "R²":     f"{r2_score(y_test, pred):.4f}",
    }
    print(f"   RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']} | R²: {metrics['R²']}")
    salvar_regressao(y_test, pred, SLUG_SVM, "SVM")
    return metrics
