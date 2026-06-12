"""
Utilitários partilhados de avaliação Spark ML (Stage 8 BD).

- Métricas globais e por classe (precision, recall, F1)
- Confusion matrix, curvas ROC/PR e pontos exportados em CSV
- Figuras padronizadas em outputs/modelos/{model_slug}/{tarefa}/
- Tuning sequencial train/val (sem multiprocessing — estável em Spark local/Docker)
"""

import os
import time
import numpy as np
import pandas as pd

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F

from spark_display import imprimir_pandas
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

from model_plots import salvar_matriz_confusao, salvar_curvas_roc_pr as mp_salvar_curvas_roc_pr
from model_plots import task_from_label_col

from .spark_setup import RANDOM_STATE, OUTPUT_DIR, should_write_reports


def make_pr_evaluator(label_col="tip_given", raw_col="rawPrediction"):
    return BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderPR"
    )


def salvar_confusion_matrix(
    df, model_slug, pred_col="prediction", label_col="tip_given", titulo_modelo=None,
):
    cm = (
        df.groupBy(label_col, pred_col).count()
        .orderBy(label_col, pred_col)
        .toPandas()
        .pivot(index=label_col, columns=pred_col, values="count")
        .fillna(0).astype(int)
    )
    if should_write_reports():
        path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_slug}.csv")
        cm.to_csv(path)
        print(f"   ✓ confusion_matrix_{model_slug}.csv")

    pred_pd = df.select(label_col, pred_col).toPandas()
    y_true = pred_pd[label_col].astype(int).values
    y_pred = pred_pd[pred_col].astype(int).values
    task = task_from_label_col(label_col)
    titulo = titulo_modelo or model_slug.replace("_", " ").title()
    salvar_matriz_confusao(y_true, y_pred, model_slug, task, titulo, label_col)

    cm_out = cm.reset_index(names="real")
    cm_out.columns = ["real"] + [f"pred_{c}" for c in cm_out.columns[1:]]
    imprimir_pandas(cm_out, dec=0, titulo="Matriz de confusão")
    return cm


def metricas_por_classe(cm, nome_modelo):
    """
    Calcula precision, recall, F1 e accuracy por classe a partir da matriz de confusão.
    Para classificação binária: classe 0 (sem gorjeta) e classe 1 (com gorjeta).
    """
    classes = sorted(cm.index.tolist())
    rows = []
    total = cm.values.sum()

    for cls in classes:
        tp = cm.loc[cls, cls] if cls in cm.columns else 0
        fp = cm[cls].sum() - tp if cls in cm.columns else 0
        fn = cm.loc[cls].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm.loc[cls].sum())
        rows.append({
            "Modelo": nome_modelo,
            "Classe": int(cls),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Support": support,
        })

    correct = sum(cm.loc[c, c] for c in classes if c in cm.columns)
    acc = correct / total if total else 0

    df_metrics = pd.DataFrame(rows)
    if should_write_reports():
        path = os.path.join(OUTPUT_DIR, f"metricas_por_classe_{nome_modelo}.csv")
        df_metrics.to_csv(path, index=False)
        print(f"   ✓ metricas_por_classe_{nome_modelo}.csv")
    imprimir_pandas(df_metrics, dec=4, titulo="Métricas por classe")
    print(f"   Accuracy global: {acc:.4f}")
    print()
    return df_metrics


def avaliar_classificador(pred_df, pred_col="prediction", raw_col="rawPrediction", label_col="tip_given"):
    """Métricas globais via Spark evaluators (Binary + Multiclass)."""
    return {
        "Accuracy":  MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=pred_col, metricName="accuracy"
        ).evaluate(pred_df),
        "Precision": MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=pred_col, metricName="weightedPrecision"
        ).evaluate(pred_df),
        "Recall": MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=pred_col, metricName="weightedRecall"
        ).evaluate(pred_df),
        "F1-score": MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol=pred_col, metricName="f1"
        ).evaluate(pred_df),
        "AUC-ROC": BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderROC"
        ).evaluate(pred_df),
        "PR-AUC": BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderPR"
        ).evaluate(pred_df),
    }


def salvar_curvas_roc_pr(
    pred_df, model_slug, prob_col=None, raw_col=None,
    label_col="tip_given", titulo_modelo=None,
):
    """
    Gera figura ROC/PR padronizada e exporta pontos das curvas em CSV (Stage 8 BD).
    prob_col: Vector de probabilidades (RF, LR).
    """
    from pyspark.ml.functions import vector_to_array

    score_col = prob_col or raw_col
    if not score_col:
        raise ValueError("Indicar prob_col ou raw_col")
    score_expr = vector_to_array(F.col(score_col)).getItem(1)
    prob_pd = pred_df.select(score_expr.alias("score"), label_col).toPandas()
    scores = prob_pd["score"].astype(float).values
    y_true = prob_pd[label_col].astype(int).values

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    prec, rec, pr_thresholds = precision_recall_curve(y_true, scores)

    if should_write_reports():
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
            os.path.join(OUTPUT_DIR, f"roc_curve_points_{model_slug}.csv"), index=False
        )
        pd.DataFrame({
            "recall": rec,
            "precision": prec,
            "threshold": list(pr_thresholds) + [np.nan],
        }).to_csv(os.path.join(OUTPUT_DIR, f"pr_curve_points_{model_slug}.csv"), index=False)
        print(f"   ✓ roc_curve_points_{model_slug}.csv")
        print(f"   ✓ pr_curve_points_{model_slug}.csv")

    task = task_from_label_col(label_col)
    titulo = titulo_modelo or model_slug.replace("_", " ").title()
    mp_salvar_curvas_roc_pr(y_true, scores, model_slug, task, titulo, label_col)


def _evaluator_metric_name(evaluator):
    try:
        return evaluator.getMetricName()
    except Exception:
        return getattr(evaluator, "metricName", "metric")


def _evaluator_maximize(evaluator):
    try:
        return bool(evaluator.isLargerBetter())
    except Exception:
        pass
    name = _evaluator_metric_name(evaluator).lower()
    return name not in ("rmse", "mse", "mae", "logloss", "logLoss")


def tunar_com_validation(estimator, param_grid, evaluator, train_prep, val_prep):
    """
    Grid search sequencial: fit em train_prep, avaliação em val_prep.

    TrainValidationSplit com parallelism usa multiprocessing.Pool, o que falha
    em Spark local[*] (Docker/Windows): TypeError SparkSession not callable.
    """
    train_count = train_prep.count()
    val_count = val_prep.count()
    n = len(param_grid)
    maximize = _evaluator_maximize(evaluator)
    metric_label = _evaluator_metric_name(evaluator)

    print(f"   Search space: {n} combinações (grid sequencial train→val)")
    print(f"   Train: {train_count:,} | Val: {val_count:,}")
    print(f"   Métrica de tuning: {metric_label} ({'↑' if maximize else '↓'} melhor)")

    best_model = None
    best_score = float("-inf") if maximize else float("inf")

    t0 = time.time()
    for i, param_map in enumerate(param_grid, 1):
        fitted = estimator.copy(param_map).fit(train_prep)
        score = evaluator.evaluate(fitted.transform(val_prep))
        improved = score > best_score if maximize else score < best_score
        mark = " <- melhor" if improved else ""
        print(f"   [{i}/{n}] {metric_label}={score:.4f}{mark}")
        if improved:
            best_score = score
            best_model = fitted

    print(f"   Treino concluído em {time.time() - t0:.1f}s | melhor {metric_label}={best_score:.4f}")
    if best_model is None:
        raise RuntimeError("Grid search não encontrou modelo válido")
    return best_model
