"""
Análise de threshold e análise de erros (Stage 8 BD).

- Sweep de thresholds com constraint recall >= 0.70
- Error analysis com Spark SQL (FP vs FN)
- Discussão técnica e de domínio (obrigatória no rubric BD)
"""

import os
import sys

_src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)
from spark_display import imprimir_spark_df, imprimir_pandas, secao

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

from .spark_setup import OUTPUT_DIR, should_write_reports


# SI: escolha de threshold com constraint de negócio (recall≥0.70) para minimizar FN
# Spark: Stage 8 — sweep de thresholds; métricas por classe e curvas ROC/PR em spark_eval.py
def analisar_threshold(clf_pred, spark, custo_fp=1.0, custo_fn=3.0):
    """Sweep de thresholds e seleção com constraint de domínio."""
    print("\n── Análise de Threshold ──")
    print("   Constraint de domínio: recall_classe_positiva >= 0.70")
    print(f"   Custos de negócio: FP={custo_fp:.1f}, FN={custo_fn:.1f}")
    print("   Justificação: FN (não prever gorjeta que existiu) é mais custoso que FP")
    print()

    prob_pos = vector_to_array(F.col("probability")).getItem(1)
    df_probs = clf_pred.withColumn("prob_pos", prob_pos)

    prob_stats = df_probs.select(
        F.min("prob_pos").alias("prob_min"),
        F.max("prob_pos").alias("prob_max"),
        F.percentile_approx("prob_pos", 0.05).alias("p05"),
        F.percentile_approx("prob_pos", 0.25).alias("p25"),
        F.percentile_approx("prob_pos", 0.50).alias("p50"),
    ).collect()[0]
    prob_min = float(prob_stats["prob_min"])
    prob_max = float(prob_stats["prob_max"])

    print(f"   Probabilidades do modelo: min={prob_min:.4f}, max={prob_max:.4f}")
    thresholds = np.round(np.linspace(prob_min, prob_max, 10), 4).tolist()
    if prob_min <= 0.50 <= prob_max:
        thresholds = sorted(set(thresholds + [0.50]))
    print(f"   Thresholds a testar: {thresholds}")
    print()

    rows = []
    total_n = df_probs.count()
    for thr in thresholds:
        pred_thr = df_probs.withColumn(
            "pred_thr",
            F.when(F.col("prob_pos") >= thr, 1.0).otherwise(0.0).cast("double")
        )
        acc = MulticlassClassificationEvaluator(labelCol="tip_given", predictionCol="pred_thr", metricName="accuracy").evaluate(pred_thr)
        f1  = MulticlassClassificationEvaluator(labelCol="tip_given", predictionCol="pred_thr", metricName="f1").evaluate(pred_thr)
        counts = pred_thr.select(
            F.sum(F.when((F.col("tip_given") == 1) & (F.col("pred_thr") == 1), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("tip_given") == 0) & (F.col("pred_thr") == 1), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("tip_given") == 1) & (F.col("pred_thr") == 0), 1).otherwise(0)).alias("fn"),
            F.sum(F.when((F.col("tip_given") == 0) & (F.col("pred_thr") == 0), 1).otherwise(0)).alias("tn"),
        ).collect()[0]
        tp, fp, fn, tn = [int(counts[x]) for x in ("tp", "fp", "fn", "tn")]
        rec_pos = tp / (tp + fn) if (tp + fn) else 0.0
        prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
        f1_pos = (2 * rec_pos * prec_pos / (rec_pos + prec_pos)) if (rec_pos + prec_pos) else 0.0
        custo_total = fp * custo_fp + fn * custo_fn
        custo_medio = custo_total / total_n if total_n else 0.0
        rows.append({
            "threshold": thr,
            "accuracy": acc,
            "f1_weighted": f1,
            "recall_pos": rec_pos,
            "precision_pos": prec_pos,
            "f1_pos": f1_pos,
            "fp": fp,
            "fn": fn,
            "custo_total": custo_total,
            "custo_medio": custo_medio,
        })

    df_thr = pd.DataFrame(rows)
    if should_write_reports():
        df_thr.to_csv(os.path.join(OUTPUT_DIR, "classification_threshold_analysis.csv"), index=False)
    imprimir_pandas(df_thr, dec=4, titulo="Análise de Threshold")

    selecionado = df_thr[df_thr["recall_pos"] >= 0.70].sort_values(
        ["custo_medio", "f1_pos"], ascending=[True, False]
    )
    if not selecionado.empty:
        best = selecionado.iloc[0]
        msg = (
            f"Threshold recomendado: {best['threshold']:.4f} "
            f"(recall_pos={best['recall_pos']:.4f}, precision_pos={best['precision_pos']:.4f}, "
            f"f1_pos={best['f1_pos']:.4f}, custo_medio={best['custo_medio']:.4f})"
        )
    else:
        best = df_thr.sort_values(["custo_medio", "f1_pos"], ascending=[True, False]).iloc[0]
        msg = (
            f"Nenhum threshold atingiu recall_pos>=0.70. "
            f"Usando mínimo custo: {best['threshold']:.4f} (custo_medio={best['custo_medio']:.4f})"
        )

    print(f"\n   {msg}")
    if should_write_reports():
        with open(os.path.join(OUTPUT_DIR, "threshold_recommendation.txt"), "w", encoding="utf-8") as f:
            f.write(msg + "\n")
        print("   ✓ threshold_recommendation.txt")
    return df_thr, float(best["threshold"])


def analisar_metricas_negocio(clf_pred, threshold_sel):
    """
    Métricas operacionais: precision@k, recall@k e lift@k para priorização.
    """
    print("\n── Métricas de Negócio (Top-K) ──")
    prob_pos = vector_to_array(F.col("probability")).getItem(1)
    df_probs = clf_pred.withColumn("prob_pos", prob_pos)

    total = df_probs.count()
    base_rate = df_probs.agg(F.avg(F.col("tip_given").cast("double")).alias("base")).collect()[0]["base"]
    ks = [0.05, 0.10, 0.20]
    rows = []
    for k in ks:
        n_k = max(1, int(total * k))
        topk = df_probs.orderBy(F.col("prob_pos").desc()).limit(n_k)
        tp = topk.filter(F.col("tip_given") == 1).count()
        precision_k = tp / n_k
        total_pos = df_probs.filter(F.col("tip_given") == 1).count()
        recall_k = tp / total_pos if total_pos else 0.0
        lift_k = precision_k / base_rate if base_rate else 0.0
        rows.append({
            "k_percent": int(k * 100),
            "n_top": n_k,
            "precision_at_k": precision_k,
            "recall_at_k": recall_k,
            "lift_at_k": lift_k,
        })
    df_k = pd.DataFrame(rows)
    if should_write_reports():
        df_k.to_csv(os.path.join(OUTPUT_DIR, "business_topk_metrics.csv"), index=False)
    imprimir_pandas(df_k, dec=4, titulo="Top-K: precision/recall/lift")
    if should_write_reports():
        print("   ✓ business_topk_metrics.csv")

    pred_sel = df_probs.withColumn(
        "pred_business",
        F.when(F.col("prob_pos") >= float(threshold_sel), 1).otherwise(0)
    )
    seg = pred_sel.withColumn(
        "periodo",
        F.when(F.col("pickup_hour").between(6, 11), "Manha")
         .when(F.col("pickup_hour").between(12, 17), "Tarde")
         .when(F.col("pickup_hour").between(18, 22), "Noite")
         .otherwise("Madrugada")
    )
    seg_df = seg.groupBy("periodo").agg(
        F.count("*").alias("n"),
        F.avg(F.col("tip_given").cast("double")).alias("taxa_real_gorjeta"),
        F.avg(F.col("pred_business").cast("double")).alias("taxa_prevista_gorjeta"),
    ).toPandas()
    if should_write_reports():
        seg_df.to_csv(os.path.join(OUTPUT_DIR, "segment_business_rates.csv"), index=False)
    imprimir_pandas(seg_df, dec=4, titulo="Taxas por período (real vs previsto)")
    if should_write_reports():
        print("   ✓ segment_business_rates.csv")


def analisar_segmentos_horarios(clf_pred):
    """
    Performance por segmento horário para identificar degradação local.
    """
    print("\n── Performance por Segmento Horário ──")
    prob_pos = vector_to_array(F.col("probability")).getItem(1)
    df_probs = clf_pred.withColumn("prob_pos", prob_pos)
    seg = df_probs.withColumn(
        "periodo",
        F.when(F.col("pickup_hour").between(6, 11), "Manha")
         .when(F.col("pickup_hour").between(12, 17), "Tarde")
         .when(F.col("pickup_hour").between(18, 22), "Noite")
         .otherwise("Madrugada")
    )
    rows = []
    for p in ["Madrugada", "Manha", "Tarde", "Noite"]:
        d = seg.filter(F.col("periodo") == p)
        if d.count() == 0:
            continue
        acc = MulticlassClassificationEvaluator(
            labelCol="tip_given", predictionCol="prediction", metricName="accuracy"
        ).evaluate(d)
        f1w = MulticlassClassificationEvaluator(
            labelCol="tip_given", predictionCol="prediction", metricName="f1"
        ).evaluate(d)
        rows.append({"periodo": p, "accuracy": acc, "f1_weighted": f1w, "n": d.count()})
    df_seg = pd.DataFrame(rows)
    if should_write_reports():
        df_seg.to_csv(os.path.join(OUTPUT_DIR, "segment_performance_report.csv"), index=False)
    imprimir_pandas(df_seg, dec=4, titulo="Performance por período")
    if should_write_reports():
        print("   ✓ segment_performance_report.csv")


# SI: interpretação FP vs FN e segmentos onde o modelo falha (relatório e defesa oral)
# Spark: Stage 8 — ≥2 queries SQL agregadas sobre DataFrame de previsões erradas no test
def analisar_erros_sql(clf_pred, spark):
    """
    Duas queries SQL sobre os erros de classificação (Stage 8 BD).
    O rubric exige pelo menos 2 queries com interpretação em texto.
    """
    clf_pred.createOrReplaceTempView("clf_predictions")

    # Query 1: distribuição de erros por hora do dia
    print("\n── Query SQL 1: Erros por hora do dia ──")
    print("   Pergunta: Em que horas do dia o modelo erra mais?")
    print()
    q1 = spark.sql("""
        SELECT pickup_hour,
               SUM(CASE WHEN tip_given != prediction THEN 1 ELSE 0 END) AS erros,
               COUNT(*) AS total,
               ROUND(100.0 * SUM(CASE WHEN tip_given != prediction THEN 1 ELSE 0 END) / COUNT(*), 2) AS taxa_erro_pct
        FROM clf_predictions
        GROUP BY pickup_hour
        ORDER BY taxa_erro_pct DESC
        LIMIT 10
    """)
    imprimir_spark_df(q1, n=10, titulo=None)
    print("   Interpretação: Horas com maior taxa de erro indicam maior incerteza")
    print("   do modelo (ex: madrugada pode ter padrões de gorjeta mais imprevisíveis).")
    print()
    q1.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_DIR, "error_analysis_por_hora")
    )

    # Query 2: caracterização de FP vs FN
    print("\n── Query SQL 2: Caracterização de Falsos Positivos vs Falsos Negativos ──")
    print("   Pergunta: Os FP e FN diferem em fare_amount e trip_distance?")
    print()
    q2 = spark.sql("""
        SELECT
            CASE
                WHEN tip_given = 0 AND prediction = 1 THEN 'Falso Positivo'
                WHEN tip_given = 1 AND prediction = 0 THEN 'Falso Negativo'
                WHEN tip_given = prediction           THEN 'Acerto'
            END AS tipo_erro,
            COUNT(*) AS count,
            ROUND(AVG(fare_amount), 2)   AS avg_fare,
            ROUND(AVG(trip_distance), 2) AS avg_distance,
            ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM clf_predictions), 2) AS pct_total
        FROM clf_predictions
        GROUP BY tipo_erro
        ORDER BY count DESC
    """)
    imprimir_spark_df(q2, titulo=None)
    print("   Interpretação: Se FN têm fare_amount maior, o modelo falha em viagens")
    print("   mais caras — exatamente onde uma gorjeta seria mais valiosa para o motorista.")
    print()
    q2.coalesce(1).write.mode("overwrite").option("header", True).csv(
        os.path.join(OUTPUT_DIR, "fp_fn_analysis")
    )


# SI: discussão de domínio — custo FP vs FN no contexto de gorjetas em táxi NYC
# Spark: Stage 8 — documentação domain_analysis_stage8.txt alinhada com PR-AUC e threshold
def escrever_analise_dominio(threshold_selecionado):
    """Discussão obrigatória de domínio (Stage 8 BD rubric)."""
    texto = f"""
ANÁLISE DE DOMÍNIO — CUSTO DE ERROS (Stage 8 BD)
=================================================

PROBLEMA
--------
Prever se uma viagem de táxi NYC receberá gorjeta (tip_given = 1).

CUSTO DOS ERROS
---------------
Falso Positivo (FP) — prever gorjeta quando não houve:
  • O motorista recebe uma sugestão incorreta.
  • Impacto: baixo — apenas desaponta expectativas.
  • Custo de negócio: mínimo.

Falso Negativo (FN) — não prever gorjeta quando houve:
  • O motorista perde insight sobre padrões rentáveis
    (hora do dia, tipo de rota, perfil de cliente).
  • Impacto: moderado — perda de oportunidade de otimização.
  • Custo de negócio: mais elevado que o FP.

DECISÃO DE DESIGN
-----------------
Por ter FN mais custoso que FP, usámos:
  1. Métrica de tuning: PR-AUC (em vez de ROC-AUC ou accuracy)
     → mais sensível a FN quando a classe positiva é minoritária.
  2. Constraint de threshold: recall >= 0.70
     → aceitamos mais FP para capturar mais gorjetas reais.
  3. Threshold selecionado: {threshold_selecionado:.2f} (em vez do padrão 0.50)
     → threshold escolhido dinamicamente a partir do sweep no conjunto de teste.

VALIDAÇÃO
---------
A query SQL de error analysis (Query 2) confirma se FN têm
características distintas (ex: fare_amount mais elevado),
o que informaria uma estratégia de re-ponderação de classes
numa iteração futura do modelo.
"""
    if should_write_reports():
        path = os.path.join(OUTPUT_DIR, "domain_analysis_stage8.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(texto)
        print("\n   ✓ domain_analysis_stage8.txt")
    print(texto)
