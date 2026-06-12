"""
Fase 2 Runner — orquestra todos os módulos de Machine Learning.

Ordem de execução:
  1. Carrega dataset e cria sessão Spark
  2. Split estratificado + pipeline Spark ML (StringIndexer + OneHotEncoder)
  3. Modelos Spark ML: RF + LR (TrainValidationSplit com train+val)
  4. Modelos sklearn: SVM, J48, MLP, Naive Bayes
  5. Repete classificação para tip_high (gorjeta alta vs baixa)
  6. Análise Stage 8: threshold, métricas por classe, ROC/PR, error analysis SQL
  7. Guarda resultados consolidados para fase 4
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fase2.spark_setup   import criar_spark, carregar_dataset, OUTPUT_DIR, FIGURES_DIR
from fase2.spark_setup   import should_write_reports
from fase2.spark_pipeline import fazer_splits, construir_pipeline
from fase2.random_forest  import treinar_random_forest_clf, treinar_random_forest_reg
from fase2.logistic_regression import treinar_logistic_regression
from fase2.svm            import treinar_svm_clf, treinar_svm_reg
from fase2.decision_tree  import treinar_decision_tree_clf, treinar_decision_tree_reg
from fase2.mlp            import treinar_mlp_clf, treinar_mlp_reg
from fase2.naive_bayes    import treinar_naive_bayes
from fase2.error_analysis import analisar_threshold, analisar_erros_sql, escrever_analise_dominio
from fase2.error_analysis import analisar_metricas_negocio, analisar_segmentos_horarios

SEP = "=" * 62


def balancear_numpy(X, y, random_state=42):
    """
    Amostragem estratificada 50/50 para conjuntos de treino NumPy.

    CORREÇÃO (feedback): SVM e MLP (sklearn) devem receber dados balanceados.
    O conjunto de TESTE nunca é balanceado — avaliação honesta com proporção real.
    Usa min(n_cls0, n_cls1) amostras de cada classe.
    """
    rng = np.random.default_rng(random_state)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    sampled = np.concatenate([
        rng.choice(idx0, size=n, replace=False),
        rng.choice(idx1, size=n, replace=False),
    ])
    rng.shuffle(sampled)
    print(f"   Balanceamento 50/50: {len(y):,} → {len(sampled):,} "
          f"({(y[sampled]==0).sum()} cls0 / {(y[sampled]==1).sum()} cls1)")
    return X[sampled], y[sampled]


def extrair_numpy(train_prep, test_prep, label_col="tip_given"):
    """Converte DataFrames Spark preparados para arrays NumPy (sklearn)."""
    print(f"\n   Convertendo dados Spark → NumPy (alvo={label_col})...")
    cols = ["features", label_col, "tip_amount"]
    train_pd = train_prep.select(*cols).toPandas()
    test_pd  = test_prep.select(*cols).toPandas()

    X_train = np.array(train_pd["features"].apply(lambda x: x.toArray()).tolist())
    y_train = train_pd[label_col].values
    y_train_reg = train_pd["tip_amount"].values

    X_test = np.array(test_pd["features"].apply(lambda x: x.toArray()).tolist())
    y_test = test_pd[label_col].values
    y_test_reg = test_pd["tip_amount"].values

    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, y_train, y_train_reg, X_test, y_test, y_test_reg, feature_names


def adicionar_pesos_classe(train_df, val_df, test_df, label_col="tip_given"):
    """Pesos inversamente proporcionais à frequência da classe."""
    counts = train_df.groupBy(label_col).count().toPandas()
    n_total = counts["count"].sum()
    n_classes = counts[label_col].nunique()
    class_weights = {
        int(row[label_col]): float(n_total / (n_classes * row["count"]))
        for _, row in counts.iterrows()
    }
    print(f"\n   Class weights ({label_col}):", class_weights)

    def apply_weights(df):
        w0 = class_weights.get(0, 1.0)
        w1 = class_weights.get(1, 1.0)
        return df.withColumn(
            "class_weight",
            F.when(F.col(label_col) == 0, F.lit(w0)).otherwise(F.lit(w1)).cast("double"),
        )

    return apply_weights(train_df), apply_weights(val_df), apply_weights(test_df)


def treinar_sklearn_classificadores(
    X_tr, y_tr, X_te, y_te, feat_names, fig_dir, fig_tree_name, tree_class_names=None,
    label_col="tip_given", model_suffix="",
):
    """SVM, J48, MLP, Naive Bayes para um alvo binário.

    O balanceamento 50/50 é aplicado UMA VEZ aqui antes de todos os modelos.
    Os modelos individuais NÃO devem re-balancear (evitar sub-amostragem dupla).
    O conjunto de TESTE permanece desequilibrado (proporção real).
    """
    # ── BALANCEAMENTO 50/50: apenas uma vez, antes de todos os classificadores ──
    print("\n── Balanceamento estratificado 50/50 (treino sklearn) ──")
    X_tr_bal, y_tr_bal = balancear_numpy(X_tr, y_tr)

    # SVM: passa já os dados balanceados (o svm.py faz subsample adicional se n > 3000)
    svm_clf_m = treinar_svm_clf(
        X_tr_bal, y_tr_bal, X_te, y_te,
        figures_dir=fig_dir, label_col=label_col, model_suffix=model_suffix,
    )
    # J48: usa class_weight='balanced' internamente; recebe dados originais (sem subsample)
    dt_clf_m, _ = treinar_decision_tree_clf(
        X_tr, y_tr, X_te, y_te, feat_names,
        class_names=tree_class_names, label_col=label_col,
    )
    # MLP: recebe dados já balanceados do runner; NÃO re-balanceia internamente
    mlp_clf_m = treinar_mlp_clf(
        X_tr_bal, y_tr_bal, X_te, y_te,
        figures_dir=fig_dir, label_col=label_col, model_suffix=model_suffix,
    )
    # Naive Bayes: recebe dados já balanceados
    nb_m = treinar_naive_bayes(
        X_tr_bal, y_tr_bal, X_te, y_te,
        figures_dir=fig_dir, label_col=label_col, model_suffix=model_suffix,
    )
    return [svm_clf_m, dt_clf_m, mlp_clf_m, nb_m]


def guardar_resultados_classificacao(
    spark_rows, sklearn_rows, spark_fname, sklearn_fname, titulo_resumo,
):
    """Grava CSVs de classificação e imprime resumo."""
    from spark_display import secao, imprimir_pandas

    spark_df = pd.DataFrame(spark_rows)
    sklearn_df = pd.DataFrame(sklearn_rows)
    if should_write_reports():
        spark_df.to_csv(os.path.join(OUTPUT_DIR, spark_fname), index=False)
        sklearn_df.to_csv(os.path.join(OUTPUT_DIR, sklearn_fname), index=False)
        print(f"\n   ✓ {spark_fname}")
        print(f"   ✓ {sklearn_fname}")

    secao(titulo_resumo, sublinhado=False)
    print("=" * 62)
    imprimir_pandas(
        spark_df[["Modelo", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]],
        titulo="Spark ML",
    )
    imprimir_pandas(
        sklearn_df[["Modelo", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]],
        titulo="sklearn",
    )
    return spark_df, sklearn_df


def guardar_resultados(
    clf_spark, lr_metrics, clf_sklearn, reg_spark, reg_sklearn,
    clf_spark_high, lr_high, clf_sklearn_high,
):
    """Consolida resultados tip_given, tip_high e regressão."""
    from spark_display import secao, imprimir_pandas

    guardar_resultados_classificacao(
        [
            {"Modelo": "RandomForest", "Tipo": "Spark ML", **clf_spark},
            {"Modelo": "LogisticRegression", "Tipo": "Spark ML", **lr_metrics},
        ],
        clf_sklearn,
        "resultados_classificacao.csv",
        "sklearn_classification_results.csv",
        "RESUMO — CLASSIFICAÇÃO tip_given",
    )
    guardar_resultados_classificacao(
        [
            {"Modelo": "RandomForest", "Tipo": "Spark ML", **clf_spark_high},
            {"Modelo": "LogisticRegression", "Tipo": "Spark ML", **lr_high},
        ],
        clf_sklearn_high,
        "resultados_classificacao_tip_high.csv",
        "sklearn_classification_results_tip_high.csv",
        "RESUMO — CLASSIFICAÇÃO tip_high (gorjeta alta vs baixa)",
    )

    for r in reg_sklearn:
        r["Tipo"] = "Sklearn"
    reg_rows = [{"Modelo": "RandomForest (Spark)", "Tipo": "Spark ML", **reg_spark}]
    reg_rows += reg_sklearn
    reg_df_out = pd.DataFrame(reg_rows)
    if should_write_reports():
        reg_df_out.to_csv(os.path.join(OUTPUT_DIR, "resultados_regressao.csv"), index=False)
        print("   ✓ resultados_regressao.csv")

    secao("RESUMO — REGRESSÃO tip_amount", sublinhado=False)
    print("=" * 62)
    imprimir_pandas(reg_df_out[["Modelo", "RMSE", "MAE", "R²"]])


def main():
    if __name__ == "__main__":
        print(f"\n{SEP}")
        print("  FASE 2 — MACHINE LEARNING")
        print(f"{SEP}")

    spark = criar_spark("NYC_Taxi_Phase2")
    df    = carregar_dataset(spark)

    # ── Tarefa 1: tip_given ──────────────────────────────────────
    train, val, test = fazer_splits(df, label_col="tip_given")
    train_prep, val_prep, test_prep, _ = construir_pipeline(train, val, test)
    train_prep, val_prep, test_prep = adicionar_pesos_classe(
        train_prep, val_prep, test_prep, label_col="tip_given"
    )

    print(f"\n{SEP}")
    print("  MODELOS SPARK ML — tip_given")
    print(SEP)
    clf_spark_metrics, clf_pred = treinar_random_forest_clf(
        train_prep, val_prep, test_prep, weight_col="class_weight"
    )
    lr_metrics = treinar_logistic_regression(
        train_prep, val_prep, test_prep, weight_col="class_weight"
    )
    reg_spark_metrics = treinar_random_forest_reg(train_prep, val_prep, test_prep)

    _, threshold_sel = analisar_threshold(clf_pred, spark)
    analisar_metricas_negocio(clf_pred, threshold_sel)
    analisar_segmentos_horarios(clf_pred)
    analisar_erros_sql(clf_pred, spark)
    escrever_analise_dominio(threshold_sel)

    X_tr, y_tr, y_tr_reg, X_te, y_te, y_te_reg, feat_names = extrair_numpy(
        train_prep, test_prep, label_col="tip_given"
    )

    print(f"\n{SEP}")
    print("  MODELOS SKLEARN — tip_given")
    print(SEP)
    svm_reg_m = treinar_svm_reg(X_tr, y_tr_reg, X_te, y_te_reg)
    dt_reg_m = treinar_decision_tree_reg(X_tr, y_tr_reg, X_te, y_te_reg)
    mlp_reg_m = treinar_mlp_reg(X_tr, y_tr_reg, X_te, y_te_reg)

    clf_sklearn = treinar_sklearn_classificadores(
        X_tr, y_tr, X_te, y_te, feat_names, FIGURES_DIR, "fig_arvore_decisao.png",
        label_col="tip_given", model_suffix="",
    )
    reg_sklearn = [svm_reg_m, dt_reg_m, mlp_reg_m]

    # ── Tarefa 2: tip_high ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  CLASSIFICAÇÃO tip_high — gorjeta alta (≥P75) vs baixa")
    print(SEP)

    train_h, val_h, test_h = fazer_splits(df, label_col="tip_high", split_prefix="tip_high_")
    train_h_prep, val_h_prep, test_h_prep, _ = construir_pipeline(train_h, val_h, test_h)
    train_h_prep, val_h_prep, test_h_prep = adicionar_pesos_classe(
        train_h_prep, val_h_prep, test_h_prep, label_col="tip_high"
    )

    clf_spark_high, _ = treinar_random_forest_clf(
        train_h_prep, val_h_prep, test_h_prep,
        weight_col="class_weight", label_col="tip_high", model_suffix="_tip_high",
    )
    lr_high = treinar_logistic_regression(
        train_h_prep, val_h_prep, test_h_prep,
        weight_col="class_weight", label_col="tip_high", model_suffix="_tip_high",
    )

    X_tr_h, y_tr_h, _, X_te_h, y_te_h, _, feat_names_h = extrair_numpy(
        train_h_prep, test_h_prep, label_col="tip_high"
    )
    clf_sklearn_high = treinar_sklearn_classificadores(
        X_tr_h, y_tr_h, X_te_h, y_te_h, feat_names_h, FIGURES_DIR,
        "fig_arvore_decisao_tip_high.png",
        tree_class_names=["Gorjeta baixa", "Gorjeta alta"],
        label_col="tip_high", model_suffix="_tip_high",
    )

    guardar_resultados(
        clf_spark_metrics, lr_metrics, clf_sklearn, reg_spark_metrics, reg_sklearn,
        clf_spark_high, lr_high, clf_sklearn_high,
    )

    spark.stop()
    print(f"\n{SEP}")
    print("  ✅ Fase 2 concluída!")
    print(SEP)


if __name__ == "__main__":
    main()
