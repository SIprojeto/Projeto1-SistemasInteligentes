"""
MLP — Multi-Layer Perceptron (sklearn).

Rede neuronal clássica com 3 camadas ocultas (128-64-32).
Usa early stopping para evitar overfitting.

NOTA: o balanceamento 50/50 é feito no fase2_runner ANTES de chamar esta função.
Não duplicar o balanceamento aqui — os dados que chegam já são balanceados.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

from model_plots import SLUG_MLP_SKLEARN, salvar_classificacao, salvar_regressao

RANDOM_STATE = 42


# SI: MLP sklearn (128-64-32) para classificação — requisito de redes neuronais clássicas no SI
# Spark: features do mesmo pipeline Spark ML; early stopping evita overfitting no test
def treinar_mlp_clf(X_train, y_train, X_test, y_test,
                    figures_dir=None, label_col="tip_given", model_suffix=""):
    print("\n" + "="*60)
    print("MLP — Classificação (128-64-32, ReLU, Adam)")
    print("="*60)

    # Os dados de treino já chegam balanceados do runner (fase2_runner.balancear_numpy).
    # NÃO balancear novamente aqui — causaria sub-amostragem excessiva em datasets pequenos.
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    print(f"   Treino recebido: {len(y_train):,} ({n0} cls0 / {n1} cls1, ratio {n0/max(n1,1):.2f}:1)")
    print(f"   Teste          : {len(y_test):,} amostras (proporção real, não balanceado)")
    print("   Parâmetros: hidden=(128,64,32), activation=relu,")
    print("   solver=adam, max_iter=300, early_stopping=True")

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", solver="adam",
        max_iter=300, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15
    )
    clf.fit(X_train, y_train)

    pred  = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo":    "MLP",
        "Accuracy":  f"{accuracy_score(y_test, pred):.4f}",
        "Precision": f"{precision_score(y_test, pred, zero_division=0):.4f}",
        "Recall":    f"{recall_score(y_test, pred, zero_division=0):.4f}",
        "F1-score":  f"{f1_score(y_test, pred, zero_division=0):.4f}",
        "AUC-ROC":   f"{roc_auc_score(y_test, proba):.4f}",
    }
    print(f"   Épocas: {clf.n_iter_} | Accuracy: {metrics['Accuracy']} | F1: {metrics['F1-score']}")

    try:
        salvar_classificacao(y_test, pred, proba, SLUG_MLP_SKLEARN, "MLP (sklearn)", label_col)
    except Exception as e:
        print(f"   ⚠ Não foi possível gerar figuras MLP: {e}")

    return metrics


# SI: MLP regressão de tip_amount para comparar com Deep MLP na fase 4
def treinar_mlp_reg(X_train, y_train, X_test, y_test):
    print("\n── MLP — Regressão (128-64-32) ──")

    reg = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", solver="adam",
        max_iter=300, random_state=RANDOM_STATE,
        early_stopping=True, validation_fraction=0.1
    )
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    metrics = {
        "Modelo": "MLP",
        "RMSE":   f"{np.sqrt(mean_squared_error(y_test, pred)):.4f}",
        "MAE":    f"{mean_absolute_error(y_test, pred):.4f}",
        "R²":     f"{r2_score(y_test, pred):.4f}",
    }
    print(f"   RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']} | R²: {metrics['R²']}")
    salvar_regressao(y_test, pred, SLUG_MLP_SKLEARN, "MLP (sklearn)")
    return metrics
