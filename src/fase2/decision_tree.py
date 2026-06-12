"""
Árvore de Decisão J48 / C4.5 (sklearn).

Usa critério de entropia (equivalente ao J48 do Weka).
Gera as mesmas figuras padronizadas que os restantes classificadores.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from model_plots import (
    SLUG_J48, salvar_classificacao, salvar_arvore_decisao, salvar_regressao,
    task_from_label_col,
)

RANDOM_STATE = 42


# SI: árvore de decisão J48/C4.5 (entropy) + figuras padronizadas em outputs/modelos/j48/
# Spark: dados preparados em Spark; treino sklearn no mesmo test set da fase 2
def treinar_decision_tree_clf(
    X_train, y_train, X_test, y_test, feature_names,
    class_names=None,
    label_col="tip_given",
):
    class_names = class_names or ["Sem gorjeta", "Com gorjeta"]
    print("\n" + "="*60)
    print("DECISION TREE J48 — Classificação (critério: entropia)")
    print("="*60)
    print("   Parâmetros: criterion=entropy, max_depth=8,")
    print("   min_samples_split=10, min_samples_leaf=5")
    print("   (equivalente ao J48/C4.5 do Weka)")

    clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=8,
        min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    pred  = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo":    "Decision Tree (J48)",
        "Accuracy":  f"{accuracy_score(y_test, pred):.4f}",
        "Precision": f"{precision_score(y_test, pred):.4f}",
        "Recall":    f"{recall_score(y_test, pred):.4f}",
        "F1-score":  f"{f1_score(y_test, pred):.4f}",
        "AUC-ROC":   f"{roc_auc_score(y_test, proba):.4f}",
    }
    print(f"   Accuracy: {metrics['Accuracy']} | F1: {metrics['F1-score']} | AUC: {metrics['AUC-ROC']}")
    print(f"   Profundidade real da árvore: {clf.get_depth()} | Nº folhas: {clf.get_n_leaves()}")

    salvar_classificacao(y_test, pred, proba, SLUG_J48, "Decision Tree (J48)", label_col)
    task = task_from_label_col(label_col)
    salvar_arvore_decisao(
        clf, feature_names, SLUG_J48, task, "Decision Tree (J48)",
        class_names=class_names,
    )

    return metrics, clf


# SI: Decision Tree regressão tip_amount (baseline interpretável na fase 4)
# Spark: features escaladas do pipeline Spark; RMSE/MAE/R² no split de teste
def treinar_decision_tree_reg(X_train, y_train, X_test, y_test):
    print("\n── Decision Tree — Regressão ──")
    print("   Parâmetros: criterion=squared_error, max_depth=8")

    reg = DecisionTreeRegressor(
        criterion="squared_error", max_depth=8,
        min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    metrics = {
        "Modelo": "Decision Tree (J48)",
        "RMSE":   f"{np.sqrt(mean_squared_error(y_test, pred)):.4f}",
        "MAE":    f"{mean_absolute_error(y_test, pred):.4f}",
        "R²":     f"{r2_score(y_test, pred):.4f}",
    }
    print(f"   RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']} | R²: {metrics['R²']}")
    salvar_regressao(y_test, pred, SLUG_J48, "Decision Tree (J48)")
    return metrics
