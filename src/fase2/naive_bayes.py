"""
Naive Bayes — Classificação (sklearn GaussianNB).

Modelo probabilístico que assume independência entre features.
Muito rápido e interpretável, serve como baseline probabilístico.
Assume que cada feature segue distribuição gaussiana por classe.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from model_plots import SLUG_NAIVE_BAYES, salvar_classificacao

RANDOM_STATE = 42


# SI: Naive Bayes (GaussianNB) — quarto algoritmo clássico obrigatório no enunciado SI
def treinar_naive_bayes(X_train, y_train, X_test, y_test,
                        figures_dir=None, label_col="tip_given", model_suffix=""):
    print("\n" + "="*60)
    print("NAIVE BAYES — Classificação (GaussianNB)")
    print("="*60)
    print("   Modelo probabilístico com assunção de independência entre features.")
    print("   Vantagens: muito rápido, interpretável, funciona bem com poucos dados.")
    print("   Limitação: a assunção de independência raramente se verifica na prática.")

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    pred  = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo":    "Naive Bayes",
        "Accuracy":  f"{accuracy_score(y_test, pred):.4f}",
        "Precision": f"{precision_score(y_test, pred, zero_division=0):.4f}",
        "Recall":    f"{recall_score(y_test, pred, zero_division=0):.4f}",
        "F1-score":  f"{f1_score(y_test, pred, zero_division=0):.4f}",
        "AUC-ROC":   f"{roc_auc_score(y_test, proba):.4f}",
    }
    print(f"   Accuracy: {metrics['Accuracy']} | F1: {metrics['F1-score']} | AUC: {metrics['AUC-ROC']}")

    print(f"\n   Probabilidades a priori aprendidas:")
    for i, prob in enumerate(clf.class_prior_):
        label = "Com gorjeta" if i == 1 else "Sem gorjeta"
        print(f"     Classe {i} ({label}): {prob:.4f}")

    try:
        salvar_classificacao(y_test, pred, proba, SLUG_NAIVE_BAYES, "Naive Bayes", label_col)
    except Exception as e:
        print(f"   ⚠ Não foi possível gerar figuras Naive Bayes: {e}")

    return metrics
