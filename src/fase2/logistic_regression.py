"""
Regressão Logística — Classificação (Spark ML).
"""

import os

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder

from model_plots import SLUG_LOGISTIC_REGRESSION

from .spark_setup import MODELS_DIR, FAST_TUNING, should_write_models
from .spark_eval import (
    make_pr_evaluator,
    salvar_confusion_matrix,
    metricas_por_classe,
    avaliar_classificador,
    salvar_curvas_roc_pr,
    tunar_com_validation,
)


# SI: baseline linear interpretável para comparar com RF e modelos sklearn
# Spark: Stage 7 — LogisticRegression + TrainValidationSplit (train+val, PR-AUC); Stage 8 — ROC/PR por classe
def treinar_logistic_regression(train_prep, val_prep, test_prep, weight_col=None,
                              label_col="tip_given", model_suffix=""):
    alvo = "tip_high" if label_col == "tip_high" else "tip_given"
    print("\n" + "=" * 60)
    print(f"LOGISTIC REGRESSION — Classificação {alvo} (Spark ML)")
    print("=" * 60)
    print("   Modelo linear com regularização ElasticNet.")

    lr_kwargs = {
        "labelCol": label_col,
        "featuresCol": "features",
        "predictionCol": "prediction_lr",
        "probabilityCol": "probability_lr",
        "rawPredictionCol": "rawPrediction_lr",
        "maxIter": 100,
    }
    if weight_col:
        lr_kwargs["weightCol"] = weight_col
        print(f"   Class weighting ativo: weightCol='{weight_col}'")
    lr = LogisticRegression(**lr_kwargs)

    if FAST_TUNING:
        param_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, [0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 1.0])
            .build()
        )
    else:
        param_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, [0.001, 0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
            .build()
        )

    best = tunar_com_validation(
        lr, param_grid, make_pr_evaluator(label_col=label_col, raw_col="rawPrediction_lr"),
        train_prep, val_prep,
    )
    model_name = f"spark_lr_classifier{model_suffix}"
    if should_write_models():
        best.write().overwrite().save(os.path.join(MODELS_DIR, model_name))
        print(f"   ✓ Melhor modelo guardado em outputs/models/{model_name}")
    else:
        print("   ℹ WRITE_MODELS=0: modelo não guardado em disco")

    pred = best.transform(test_prep).cache()
    cm = salvar_confusion_matrix(
        pred, SLUG_LOGISTIC_REGRESSION, "prediction_lr", label_col=label_col,
        titulo_modelo="Logistic Regression",
    )
    metricas_por_classe(cm, SLUG_LOGISTIC_REGRESSION)
    metrics = avaliar_classificador(
        pred, pred_col="prediction_lr", raw_col="rawPrediction_lr", label_col=label_col
    )

    print("\n── Métricas globais no conjunto de teste ──")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    salvar_curvas_roc_pr(
        pred, SLUG_LOGISTIC_REGRESSION, prob_col="probability_lr",
        label_col=label_col, titulo_modelo="Logistic Regression",
    )
    return metrics
