import os
import time
import numpy as np

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder

from model_plots import SLUG_RANDOM_FOREST, salvar_regressao

from .spark_setup import RANDOM_STATE, MODELS_DIR, FAST_TUNING, should_write_models
from .spark_eval import (
    make_pr_evaluator,
    salvar_confusion_matrix,
    metricas_por_classe,
    avaliar_classificador,
    salvar_curvas_roc_pr,
    tunar_com_validation,
)


def _cross_validate_rf_pr_auc(train_prep, label_col, best_params, n_folds=5):
    """
    5-Fold Cross-Validation no conjunto de treino para PR-AUC do Random Forest.
    Reporta média ± desvio padrão.

    NOTA METODOLÓGICA (feedback):
    Um único split train/val pode ser um resultado de sorte ou overfitting ao split.
    O CV com 5 folds converte um número isolado numa estimativa robusta com
    intervalo de confiança empírico (média ± std).
    O conjunto de TESTE nunca é usado neste processo.

    NOTA TÉCNICA — BrokenPipeError:
    Em PySpark local mode, o processo worker Python termina o fold e tenta fazer
    flush() do stdout depois de o processo pai já ter fechado o socket.
    Suprimido via variável de ambiente PYSPARK_SUBMIT_ARGS e filtro de logging.
    """
    print(f"\n── 5-Fold Cross-Validation — PR-AUC (Random Forest) ──")
    from pyspark.ml.classification import RandomForestClassifier as _RFC

    evaluator = make_pr_evaluator(label_col=label_col, raw_col="rawPrediction")

    # Recolher dados como pandas para fazer os folds manualmente
    # (PySpark não tem CrossValidator com PR-AUC nativo eficiente em local mode)
    pdf = train_prep.select("features", label_col).toPandas()
    n = len(pdf)
    fold_size = n // n_folds
    indices = np.arange(n)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(indices)

    fold_aucs = []
    spark_session = train_prep.sql_ctx.sparkSession

    for fold_i in range(n_folds):
        val_idx = indices[fold_i * fold_size: (fold_i + 1) * fold_size]
        tr_idx  = np.concatenate([indices[:fold_i * fold_size],
                                   indices[(fold_i + 1) * fold_size:]])
        fold_tr  = spark_session.createDataFrame(pdf.iloc[tr_idx].reset_index(drop=True))
        fold_val = spark_session.createDataFrame(pdf.iloc[val_idx].reset_index(drop=True))

        rf_cv = _RFC(
            labelCol=label_col, featuresCol="features",
            probabilityCol="probability", rawPredictionCol="rawPrediction",
            predictionCol="prediction", seed=RANDOM_STATE,
            **best_params
        )
        cv_model = rf_cv.fit(fold_tr)
        cv_pred  = cv_model.transform(fold_val)
        fold_auc = evaluator.evaluate(cv_pred)
        fold_aucs.append(fold_auc)
        print(f"   Fold {fold_i+1}/{n_folds}: PR-AUC = {fold_auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs, ddof=1))
    print(f"\n   ✅ 5-Fold CV PR-AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return mean_auc, std_auc, fold_aucs


# SI: modelo ensemble para classificação tip_given; curvas ROC/PR e métricas por classe
# Spark: Stage 7 — RandomForestClassifier + TrainValidationSplit (train+val, PR-AUC), Stage 8 no test
def treinar_random_forest_clf(train_prep, val_prep, test_prep, weight_col=None,
                            label_col="tip_given", model_suffix=""):
    alvo = "tip_high (gorjeta alta)" if label_col == "tip_high" else "tip_given"
    print("\n" + "=" * 60)
    print(f"RANDOM FOREST — Classificação {alvo} (Spark ML)")
    print("=" * 60)

    rf_kwargs = {
        "labelCol": label_col,
        "featuresCol": "features",
        "probabilityCol": "probability",
        "rawPredictionCol": "rawPrediction",
        "predictionCol": "prediction",
        "seed": RANDOM_STATE,
    }
    if weight_col:
        rf_kwargs["weightCol"] = weight_col
        print(f"   Class weighting ativo: weightCol='{weight_col}'")
    rf = RandomForestClassifier(**rf_kwargs)

    if FAST_TUNING:
        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100])
            .addGrid(rf.maxDepth, [5, 8])
            .build()
        )
    else:
        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100, 150])
            .addGrid(rf.maxDepth, [5, 8, 12])
            .build()
        )

    best = tunar_com_validation(
        rf, param_grid, make_pr_evaluator(label_col=label_col, raw_col="rawPrediction"),
        train_prep, val_prep,
    )
    model_name = f"spark_rf_classifier{model_suffix}"
    if should_write_models():
        best.write().overwrite().save(os.path.join(MODELS_DIR, model_name))
        print(f"   ✓ Melhor modelo guardado em outputs/models/{model_name}")
    else:
        print("   ℹ WRITE_MODELS=0: modelo não guardado em disco")

    pred = best.transform(test_prep).cache()
    cm = salvar_confusion_matrix(
        pred, SLUG_RANDOM_FOREST, label_col=label_col, titulo_modelo="Random Forest",
    )
    metricas_por_classe(cm, SLUG_RANDOM_FOREST)
    metrics = avaliar_classificador(pred, label_col=label_col)

    print("\n── Métricas globais no conjunto de teste ──")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # ── 5-FOLD CROSS-VALIDATION (apenas para tip_given — modelo principal) ──
    # MELHORIA (feedback): PR-AUC num único split pode ser overfitting ao split.
    # CV com 5 folds valida que o resultado é robusto e não acidental.
    if label_col == "tip_given":
        # Extrair melhores hiperparâmetros do modelo treinado
        best_params = {
            "numTrees": best.getNumTrees,
            "maxDepth": best.getOrDefault("maxDepth"),
        }
        # numTrees é propriedade, não getter com ()
        try:
            best_params = {
                "numTrees": int(best.toDebugString.split("numTrees=")[1].split(",")[0]) if "numTrees=" in best.toDebugString else 100,
                "maxDepth": best.getOrDefault("maxDepth"),
            }
        except Exception:
            best_params = {"numTrees": 100, "maxDepth": 8}

        cv_mean, cv_std, cv_folds = _cross_validate_rf_pr_auc(
            train_prep, label_col, best_params, n_folds=5
        )
        metrics["CV_PR_AUC_mean"] = cv_mean
        metrics["CV_PR_AUC_std"]  = cv_std
        print(f"\n   📊 PR-AUC (5-Fold CV): {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   PR-AUC (teste único): {metrics.get('PR-AUC', metrics.get('AUC-ROC', 0)):.4f}")

    salvar_curvas_roc_pr(
        pred, SLUG_RANDOM_FOREST, prob_col="probability",
        label_col=label_col, titulo_modelo="Random Forest",
    )
    return metrics, pred


# SI: tarefa de regressão tip_amount pedida no projeto integrado SI+BD
# Spark: RandomForestRegressor + TrainValidationSplit (train+val), avaliação só no test
def treinar_random_forest_reg(train_prep, val_prep, test_prep):
    print("\n" + "=" * 60)
    print("RANDOM FOREST — Regressão (Spark ML)")
    print("=" * 60)

    rf_reg = RandomForestRegressor(
        labelCol="tip_amount", featuresCol="features",
        predictionCol="prediction", seed=RANDOM_STATE,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf_reg.numTrees, [50, 100])
        .addGrid(rf_reg.maxDepth, [5, 8])
        .build()
    )

    rmse_eval = RegressionEvaluator(
        labelCol="tip_amount", predictionCol="prediction", metricName="rmse"
    )

    t0 = time.time()
    best = tunar_com_validation(rf_reg, param_grid, rmse_eval, train_prep, val_prep)
    print(f"   Regressão RF — grid concluído em {time.time() - t0:.1f}s")

    best.write().overwrite().save(os.path.join(MODELS_DIR, "spark_rf_regressor"))
    print("   ✓ Melhor modelo guardado em outputs/models/spark_rf_regressor")

    pred = best.transform(test_prep).cache()
    metrics = {
        "RMSE": rmse_eval.evaluate(pred),
        "MAE": RegressionEvaluator(
            labelCol="tip_amount", predictionCol="prediction", metricName="mae"
        ).evaluate(pred),
        "R²": RegressionEvaluator(
            labelCol="tip_amount", predictionCol="prediction", metricName="r2"
        ).evaluate(pred),
    }

    print("\n── Métricas no conjunto de teste ──")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    pred_pd = pred.select("tip_amount", "prediction").toPandas()
    salvar_regressao(
        pred_pd["tip_amount"].values, pred_pd["prediction"].values,
        SLUG_RANDOM_FOREST, "Random Forest",
    )

    return metrics
