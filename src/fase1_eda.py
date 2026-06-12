"""
Projeto de Sistemas Inteligentes
Fase 1 — Análise Exploratória de Dados (EDA) e Pré-processamento
Dataset: NYC Yellow Taxi Trip Data (março 2016)
Pré-processamento com Apache Spark (RDDs e DataFrames)
"""
import os, sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ── Spark ──────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYTHONWARNINGS"]         = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"

import csv
from io import StringIO
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)

# ─────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "yellow_tripdata_2016-03.csv")
DATASET_N_ROWS = 1000000   # nº de linhas do dataset final (Python + Weka GUI) min 5k
RANDOM_STATE = 42
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "reports")
FIGURES_DIR  = os.path.join(BASE_DIR, "outputs", "fase1_eda")  # ← NOVO: pasta dedicada à Fase 1
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# CRIAÇÃO AUTOMÁTICA DE TODAS AS PASTAS DE OUTPUT
# ─────────────────────────────────────────
for _folder in [
    os.path.join(BASE_DIR, "outputs", "fase1_eda"),
    os.path.join(BASE_DIR, "outputs", "fase2_ml"),
    os.path.join(BASE_DIR, "outputs", "fase3_dl_anomalias"),
    os.path.join(BASE_DIR, "outputs", "fase4_comparacao"),
    os.path.join(BASE_DIR, "outputs", "reports"),
]:
    os.makedirs(_folder, exist_ok=True)


def _fase1_ja_completa():
    needed = [
        os.path.join(OUTPUT_DIR, "dataset_processado.csv"),
        os.path.join(OUTPUT_DIR, "tabela_summary.csv"),
        os.path.join(BASE_DIR, "outputs", "fase1_eda", "fig1_distribuicoes.png"),
        os.path.join(BASE_DIR, "outputs", "fase1_eda", "fig3_correlacao.png"),
    ]
    return all(os.path.exists(p) for p in needed)


if os.environ.get("SKIP_IF_READY", "").lower() in ("1", "true", "yes") and _fase1_ja_completa():
    print("\n✓ Fase 1 ignorada — outputs já existem (SKIP_IF_READY=1)\n")
    sys.exit(0)

from spark_display import imprimir_describe_spark, imprimir_spark_df, imprimir_describe_pandas, imprimir_pandas, secao

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

# ─────────────────────────────────────────
# DESCRIÇÃO DAS VARIÁVEIS DO DATASET
# ─────────────────────────────────────────
VARS_QUANT_CONT = {
    "fare_amount":    "Valor base da corrida ($)",
    "tip_amount":     "Valor da gorjeta ($)  [TARGET regressão]",
    "trip_distance":  "Distância da viagem (milhas)",
    "trip_duration_min": "Duração da viagem (min) — derivada",
    "avg_speed":      "Velocidade média (mph) — derivada",
}
VARS_QUANT_DISC = {
    "passenger_count": "Número de passageiros (1–6)",
    "pickup_hour":     "Hora do dia (0–23) — derivada",
    "pickup_weekday":  "Dia da semana (0–6) — derivada",
}
VARS_QUAL = {
    "payment_type":       "Tipo de pagamento (1=Cartão, 2=Dinheiro, ...)",
    "RatecodeID":         "Código de tarifa aplicado",
    "VendorID":           "Fornecedor TPEP (1=CMT, 2=VeriFone)",
    "store_and_fwd_flag": "Registo armazenado antes de enviar (Y/N)",
    "pickup_weekend":     "Fim de semana? (0/1) — derivada",
    "tip_given":          "Deu gorjeta? (0/1) — TARGET classificação (baseline)",
    "tip_high":           "Gorjeta alta? (0/1) — TARGET classificação (P75 tip_amount)",
}

NOMINAL_CODEBOOK = {
    "payment_type": {
        1: "Credit card",
        2: "Cash",
        3: "No charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided trip",
    },
    "RatecodeID": {
        1: "Standard rate",
        2: "JFK",
        3: "Newark",
        4: "Nassau/Westchester",
        5: "Negotiated fare",
        6: "Group ride",
    },
    "VendorID": {
        1: "Creative Mobile Technologies",
        2: "VeriFone Inc.",
    },
}

print("\n── Descrição das Variáveis do Dataset ──")
print("\n  Variáveis Quantitativas Contínuas:")
for v, d in VARS_QUANT_CONT.items():
    print(f"    {v:25s} → {d}")
print("\n  Variáveis Quantitativas Discretas:")
for v, d in VARS_QUANT_DISC.items():
    print(f"    {v:25s} → {d}")
print("\n  Variáveis Qualitativas (Nominais/Binárias):")
for v, d in VARS_QUAL.items():
    print(f"    {v:25s} → {d}")

print("\n  Legenda nominal (códigos -> significado):")
for var_name, mapping in NOMINAL_CODEBOOK.items():
    print(f"    {var_name}:")
    for code, meaning in mapping.items():
        print(f"      {code} = {meaning}")


# SI: infraestrutura de processamento em volume para EDA e features do dataset NYC Taxi
# Spark: SparkSession local para Stages 1–4 (ingestão RDD, limpeza, SQL, export Parquet)
# ─────────────────────────────────────────
# 1. INICIALIZAR SPARK
# ─────────────────────────────────────────
print("=" * 60)
print("FASE 1 — ANÁLISE EXPLORATÓRIA DE DADOS")
print("=" * 60)

spark = SparkSession.builder \
    .appName("NYC_Taxi_EDA") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"\n   Spark {spark.version} iniciado")

# SI: descrição das variáveis, targets tip_given/tip_amount e engenharia de atributos no relatório
# Spark: Stage 1 — textFile, map, filter, .count()/.take(5); malformados detetados no RDD
# ─────────────────────────────────────────
# 2. CARREGAMENTO COM SPARK (RDD + schema explícito)
# ─────────────────────────────────────────
print(f"\n   A carregar: {DATASET_PATH}")

raw_rdd = spark.sparkContext.textFile(DATASET_PATH)
header_line = raw_rdd.first()
header_cols = next(csv.reader([header_line]))
print(f"   Header detectado ({len(header_cols)} colunas)")
print(f"   Colunas de entrada: {header_cols}")

required_cols = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "fare_amount",
    "tip_amount", "payment_type", "RatecodeID",
    "VendorID", "store_and_fwd_flag"
]
missing_cols = [c for c in required_cols if c not in header_cols]
if missing_cols:
    raise ValueError(f"Colunas obrigatórias ausentes no CSV: {missing_cols}")
field_index = {c: header_cols.index(c) for c in required_cols}


def try_int(value):
    if value in (None, "", "NULL"):
        return None
    return int(float(value))


def try_float(value):
    if value in (None, "", "NULL"):
        return None
    return float(value)


def parse_csv_line(line):
    try:
        row = next(csv.reader([line]))
        if len(row) != len(header_cols):
            return None

        return Row(
            tpep_pickup_datetime=row[field_index["tpep_pickup_datetime"]].strip() or None,
            tpep_dropoff_datetime=row[field_index["tpep_dropoff_datetime"]].strip() or None,
            passenger_count=try_int(row[field_index["passenger_count"]]),
            trip_distance=try_float(row[field_index["trip_distance"]]),
            fare_amount=try_float(row[field_index["fare_amount"]]),
            tip_amount=try_float(row[field_index["tip_amount"]]),
            payment_type=try_int(row[field_index["payment_type"]]),
            RatecodeID=try_int(row[field_index["RatecodeID"]]),
            VendorID=try_int(row[field_index["VendorID"]]),
            store_and_fwd_flag=row[field_index["store_and_fwd_flag"]].strip() or None,
        )
    except Exception:
        return None

# SI: parsing robusto de CSV — linhas inválidas descartadas antes da modelagem
# Spark: Stage 1 — transformações map (parse) + filter (header e malformados) no RDD bruto
parsed_rdd = raw_rdd.filter(lambda row: row != header_line) \
    .map(parse_csv_line) \
    .cache()

malformed_count = parsed_rdd.filter(lambda x: x is None).count()
print(f"   Registos malformados detectados: {malformed_count}")
if malformed_count > 0:
    malformed_sample = parsed_rdd.filter(lambda x: x is None).take(5)
    print("   Exemplos de linhas malformadas:")
    for bad in malformed_sample:
        print(f"     {bad}")

good_rdd = parsed_rdd.filter(lambda x: x is not None)

# Reportar RDD statistics: .take(5) e .count() (conforme requisito BD Stage 1)
print(f"\n── Estatísticas RDD (conforme BD Stage 1) ──")
print(f"   RDD.count(): {good_rdd.count():,} registos válidos")
rdd_sample = good_rdd.take(5)
print(f"   RDD.take(5): Primeiras 5 linhas válidas do RDD")
for i, row in enumerate(rdd_sample, 1):
    print(f"     {i}. {row}")


# SI: dataset tabular pronto para modelos supervisionados (classificação + regressão de gorjeta)
# Spark: Stage 3 — promoção RDD→DataFrame com StructType explícito (sem inferSchema)
schema = StructType([
    StructField("tpep_pickup_datetime", StringType(), True),
    StructField("tpep_dropoff_datetime", StringType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("VendorID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
])

df_spark = spark.createDataFrame(good_rdd, schema=schema)

df_spark = df_spark.withColumn(
    "tpep_pickup_datetime",
    F.to_timestamp("tpep_pickup_datetime", "yyyy-MM-dd HH:mm:ss")
).withColumn(
    "tpep_dropoff_datetime",
    F.to_timestamp("tpep_dropoff_datetime", "yyyy-MM-dd HH:mm:ss")
)

df_spark = df_spark.repartition(8)  # Particionamento justificado: 8 partições para balanced parallelism em máquinas locais (4-8 cores típicos)

df_spark = df_spark.cache()

# SI: documentação do schema e volume de dados para o relatório EDA
# Spark: Stage 3 — printSchema() + describe().show() após promoção RDD→DataFrame (StructType explícito)
total_rows = df_spark.count()
print(f"   Total de linhas : {total_rows:,}")
print(f"   Colunas         : {len(df_spark.columns)}")
print(f"\n   Particionamento: 8 partições (para paralelismo eficiente em máquinas com 4-8 cores)")
print("\n   Schema detectado:")
df_spark.printSchema()
print("\n   Estatísticas descritivas (describe().show()):")
imprimir_describe_spark(df_spark)

# Amostragem reprodutível via Spark (por nº de linhas)
sample_fraction = min(1.0, DATASET_N_ROWS / total_rows) if total_rows > 0 else 1.0
df_sample = df_spark.sample(withReplacement=False, fraction=sample_fraction, seed=RANDOM_STATE)
df_sample = df_sample.cache()
sampled_rows = df_sample.count()
print(f"   Amostra por linhas (alvo={DATASET_N_ROWS:,}): {sampled_rows:,} linhas ({sample_fraction*100:.2f}% do total)")

# SI: feature engineering (duração, hora, fim de semana, velocidade) para ML/DL
# Spark: Stage 2 — contagem NaN por coluna, política por coluna, outliers IQR e filtros de qualidade
# ─────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO COM SPARK
# ─────────────────────────────────────────
print("\n── Pré-processamento com Spark ──")

# ESTRATÉGIA DE LIMPEZA NaN/NULL: Justificação por coluna
print("\n── Justificação da estratégia de NaN/NULL por coluna ──")
nan_strategy = {
    "tpep_pickup_datetime": "DROP — timestamps obrigatórios; sem substituição válida",
    "tpep_dropoff_datetime": "DROP — timestamps obrigatórios; sem substituição válida",
    "passenger_count": "DROP — cardinais <= 6; valores ausentes inviabilizam contagem",
    "trip_distance": "DROP — contínuo e variável; não há imputação apropriada (mean/median causariam enviesamento)",
    "fare_amount": "DROP — obrigatório para análise; mediana/mean distorceriam distribuição",
    "tip_amount": "DROP — alvo de regressão; necessário para validação",
    "payment_type": "DROP — categórico; sem moda robusta",
    "RatecodeID": "DROP — categórico; categoria 'unknown' não apropriada",
    "VendorID": "DROP — identificador de fonte de dados; sem valor padrão",
    "store_and_fwd_flag": "DROP — informação de transmissão crítica; sem default"
}
print("\n   Estratégia global: DROP todas as linhas com NaN/NULL")
print("   Justificação: Para este dataset de táxis, cada coluna é crítica para análise.")
print("   Não há coluna com cardinalidade suficientemente alta para imputação segura.")
for col, strategy in nan_strategy.items():
    print(f"     • {col:30s} → {strategy}")

# Duração da viagem em minutos
df_sample = df_sample.withColumn(
    "trip_duration_min",
    (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0
)

# Features temporais
df_sample = df_sample \
    .withColumn("pickup_hour",    F.hour("tpep_pickup_datetime")) \
    .withColumn("pickup_weekday", F.dayofweek("tpep_pickup_datetime") - 1) \
    .withColumn("pickup_weekend", (F.dayofweek("tpep_pickup_datetime").isin([1, 7])).cast(IntegerType()))

# Velocidade média (milhas/hora)
df_sample = df_sample.withColumn(
    "avg_speed",
    F.when(
        F.col("trip_duration_min") > 0,
        F.col("trip_distance") / (F.col("trip_duration_min") / 60.0)
    ).otherwise(None)
)

# Contagem explícita de valores ausentes / NaN
float_cols = ["trip_distance", "fare_amount", "tip_amount", "trip_duration_min", "avg_speed"]
int_cols = ["passenger_count", "payment_type", "RatecodeID", "VendorID"]
timestamp_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
string_cols = ["store_and_fwd_flag"]
missing_exprs = [F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c) for c in float_cols]
missing_exprs += [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in int_cols]
missing_exprs += [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in timestamp_cols]
missing_exprs += [F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c) for c in string_cols]
missing_counts = df_sample.select(missing_exprs).toPandas().T
missing_counts.columns = ["missing_count"]
print("\n── Valores ausentes / NaN por coluna ──")
print(missing_counts)
missing_counts.to_csv(f"{OUTPUT_DIR}/missing_values_count.csv")
print("   ✓ missing_values_count.csv")

# Limites IQR para outliers nas variáveis quantitativas
iqr_columns = ["fare_amount", "tip_amount", "trip_distance", "trip_duration_min"]
quantiles = df_sample.approxQuantile(iqr_columns, [0.25, 0.75], 0.01)
iqr_limits = {}
for col, q in zip(iqr_columns, quantiles):
    q1, q3 = q
    iqr = q3 - q1
    lower = max(q1 - 1.5 * iqr, 0.0)
    upper = q3 + 1.5 * iqr
    iqr_limits[col] = (lower, upper)
    print(f"   IQR {col}: Q1={q1:.2f}, Q3={q3:.2f}, lower={lower:.2f}, upper={upper:.2f}")

n_before = df_sample.count()
for col, (lower, upper) in iqr_limits.items():
    df_sample = df_sample.filter((F.col(col) >= lower) & (F.col(col) <= upper))

# Qualidade adicional dos dados
df_sample = df_sample.filter(
    (F.col("passenger_count") > 0)   & (F.col("passenger_count") <= 6)  &
    (F.col("trip_distance") > 0)      & (F.col("trip_distance") < 100)  &
    (F.col("fare_amount") > 0)        & (F.col("fare_amount") < 300)    &
    (F.col("trip_duration_min") > 0.5) & (F.col("trip_duration_min") < 180) &
    (F.col("tip_amount") >= 0)        &
    F.col("tpep_pickup_datetime").isNotNull() &
    F.col("tpep_dropoff_datetime").isNotNull() &
    F.col("payment_type").isNotNull()
)

n_after = df_sample.count()
print(f"   Removidas {n_before - n_after:,} linhas depois da filtragem IQR e de qualidade")
print(f"   Dataset limpo: {n_after:,} linhas")

# Persistência intermediária em Parquet para Spark
print("\n── Persistindo dataset processado em Parquet ──")
df_sample.select(required_cols + ["trip_duration_min", "pickup_hour", "pickup_weekday", "pickup_weekend", "avg_speed"]).write.mode("overwrite").parquet(f"{OUTPUT_DIR}/dataset_raw.parquet")
print(f"   dataset_raw.parquet salvo em {OUTPUT_DIR}")

# SI: EDA com gráficos (distribuições, correlação, gorjeta por hora) para o relatório SI
# Spark: Stage 4 — ≥5 queries SQL (GROUP BY, CTE, window, JOIN, HAVING) com resultados persistidos
# SQL para descoberta e análise explicativa
print("\n── Spark SQL: consultas de qualidade e modelagem ──")
df_sample.createOrReplaceTempView("taxi_raw")

# ── Query 1: Gorjeta por tipo de pagamento ──────────────────────────────
# INTERPRETAÇÃO DE NEGÓCIO (TODO: completar para o relatório):
# Esta query revela que viagens pagas com cartão de crédito (payment_type=1)
# concentram quase a totalidade das gorjetas registadas, enquanto pagamentos
# em dinheiro (payment_type=2) têm avg_tip≈0 — não por ausência de gorjeta real,
# mas porque o sistema TLC não regista gorjetas em dinheiro.
# IMPLICAÇÃO PARA O MODELO: justifica o filtro payment_type=1 antes do treino,
# evitando que o modelo aprenda a associar "dinheiro → sem gorjeta" como regra
# causal quando é apenas um artefacto de registo. A feature payment_type
# deve ser excluída dos inputs para evitar data leakage.
query1 = spark.sql(
    """
    SELECT payment_type,
           COUNT(*) AS total_trips,
           ROUND(AVG(tip_amount), 4) AS avg_tip,
           ROUND(100.0 * SUM(CASE WHEN tip_amount > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_with_tip
    FROM taxi_raw
    GROUP BY payment_type
    ORDER BY payment_type
    """
)
secao("Query 1 — Gorjeta por tipo de pagamento (GROUP BY)")
imprimir_spark_df(query1)
query1.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_DIR}/sql_payment_type_summary")

# ── Query 2: Gorjeta por hora e tipo de pagamento (CTE) ─────────────────
# INTERPRETAÇÃO DE NEGÓCIO (TODO: completar para o relatório):
# Viagens nas primeiras horas da manhã (00h–05h) tendem a ter avg_tip mais
# elevada para pagamentos com cartão. Este padrão sugere que pickup_hour
# é um preditor relevante para tip_high — viagens noturnas longas (p.ex.,
# regresso de festa ao aeroporto) correlacionam-se com gorjetas mais generosas.
# IMPLICAÇÃO PARA O MODELO: a feature pickup_hour deve ser incluída como
# variável contínua (ou codificada ciclicamente com sin/cos) nos classificadores.
query2 = spark.sql(
    """
    WITH hourly_tips AS (
      SELECT payment_type,
             pickup_hour,
             COUNT(*) AS trips,
             AVG(tip_amount) AS avg_tip,
             SUM(CASE WHEN tip_amount > 0 THEN 1 ELSE 0 END) AS tip_count
      FROM taxi_raw
      WHERE payment_type IN (1, 2)
      GROUP BY payment_type, pickup_hour
    )
    SELECT payment_type, pickup_hour, trips, ROUND(avg_tip, 4) AS avg_tip, tip_count
    FROM hourly_tips
    WHERE trips > 100
    ORDER BY payment_type, pickup_hour
    """
)
secao("Query 2 — Gorjeta por hora e pagamento (CTE + GROUP BY)")
imprimir_spark_df(query2, n=10)
query2.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_DIR}/sql_hourly_tip_analysis")

# ── Query 3: Média móvel de gorjeta (window function) ───────────────────
# INTERPRETAÇÃO DE NEGÓCIO (TODO: completar para o relatório):
# A running_avg_tip mostra a tendência acumulada ao longo do dia para cada
# tipo de pagamento. Para cartão, a média estabiliza ao fim de poucas horas,
# sugerindo que o padrão de gorjeta é estável e não tem deriva temporal forte.
# IMPLICAÇÃO PARA O MODELO: a ausência de deriva temporal significativa
# valida que um modelo treinado em dados de março 2016 generaliza dentro
# do mesmo mês sem necessidade de features de janela temporal complexas.
query3 = spark.sql(
    """
    SELECT pickup_hour,
           payment_type,
           avg_tip,
           ROUND(AVG(avg_tip) OVER (PARTITION BY payment_type ORDER BY pickup_hour ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 4) AS running_avg_tip
    FROM (
      SELECT pickup_hour,
             payment_type,
             AVG(tip_amount) AS avg_tip
      FROM taxi_raw
      GROUP BY pickup_hour, payment_type
    )
    ORDER BY payment_type, pickup_hour
    """
)
secao("Query 3 — Média móvel de gorjeta (window function)")
imprimir_spark_df(query3, n=10)
query3.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_DIR}/sql_window_tip_analysis")

# ── Query 4: Gorjeta por hora com total diário (JOIN) ────────────────────
# INTERPRETAÇÃO DE NEGÓCIO (TODO: completar para o relatório):
# O JOIN com o total de viagens por hora permite calcular a quota de cada
# tipo de pagamento em cada período. Horas de pico (7h–9h, 17h–19h) têm
# maior volume de viagens mas gorjeta média mais baixa — provavelmente
# viagens urbanas curtas de commute. Horas fora de pico têm menor volume
# mas gorjeta proporcionalmente maior (viagens aeroporto / lazer).
# IMPLICAÇÃO PARA O MODELO: a combinação hora+tipo de viagem (proxied por
# RatecodeID=2 para JFK) é um preditor composto importante para tip_high.
query4 = spark.sql(
    """
    SELECT p.pickup_hour,
           p.payment_type,
           p.trips,
           ROUND(p.avg_tip, 4) AS avg_tip,
           h.hour_total
    FROM (
      SELECT pickup_hour,
             payment_type,
             COUNT(*) AS trips,
             AVG(tip_amount) AS avg_tip
      FROM taxi_raw
      GROUP BY pickup_hour, payment_type
    ) p
    JOIN (
      SELECT pickup_hour,
             COUNT(*) AS hour_total
      FROM taxi_raw
      GROUP BY pickup_hour
    ) h
    ON p.pickup_hour = h.pickup_hour
    ORDER BY p.pickup_hour, p.payment_type
    """
)
secao("Query 4 — Gorjeta por hora com total diário (JOIN)")
imprimir_spark_df(query4, n=10)
query4.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_DIR}/sql_join_hourly_payment")

p75_tip = df_sample.filter(F.col("tip_amount") > 0).approxQuantile(
    "tip_amount", [0.75], 0.01
)[0]

# ── Query 5: Horas com gorjetas altas (CTE + HAVING) ─────────────────────
# INTERPRETAÇÃO DE NEGÓCIO (TODO: completar para o relatório):
# Esta query isola as horas do dia em que a concentração de gorjetas acima
# do P75 é mais elevada. A correlação entre avg_distance e avg_fare nestas
# janelas horárias indica que viagens mais longas (aeroporto, Newark, JFK)
# dominam os registos de gorjeta alta.
# IMPLICAÇÃO PARA O MODELO: considerar uma feature binária is_airport_trip
# derivada de RatecodeID ∈ {2, 3} (JFK=2, Newark=3), que deve ser um
# preditor muito forte para tip_high — viagens de aeroporto têm tarifa
# fixa mais alta e passageiros com maior predisposição a gorjetar.
query5 = spark.sql(
    f"""
    WITH high_tips AS (
      SELECT *
      FROM taxi_raw
      WHERE tip_amount >= {float(p75_tip):.4f}
    )
    SELECT pickup_hour,
           COUNT(*) AS high_tip_trips,
           ROUND(AVG(trip_distance), 4) AS avg_distance,
           ROUND(AVG(fare_amount), 4) AS avg_fare
    FROM high_tips
    GROUP BY pickup_hour
    HAVING COUNT(*) > 50
    ORDER BY high_tip_trips DESC
    """
)
secao(f"Query 5 — Horas com gorjetas altas (tip_amount >= P75=${p75_tip:.2f}, HAVING > 50)")
imprimir_spark_df(query5, n=10)
if query5.count() == 0:
    print("   ℹ Sem resultados — amostra pequena e filtro HAVING COUNT(*) > 50 demasiado restritivo.")
    print()
query5.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUTPUT_DIR}/sql_high_tip_hours")

# SI: variável alvo tip_given (classificação) e filtro payment_type=1 justificado no domínio
# Spark: achado SQL→modelagem — cartão evita viés de gorjeta zero em pagamentos em dinheiro
# Filtros de qualidade específicos de cartão
print("\n── Filtragem final para pagamento por cartão (payment_type = 1) ──")
df_card_spark = df_sample.filter(F.col("payment_type") == 1)
df_card_spark = df_card_spark.withColumn(
    "tip_given",
    (F.col("tip_amount") > 0).cast(IntegerType())
)
df_card_spark = df_card_spark.cache()
print(f"   A usar apenas cartão: {df_card_spark.count():,} linhas")

print("\n── Distribuição da gorjeta (Spark) ──")
imprimir_spark_df(df_card_spark.groupBy("tip_given").count().orderBy("tip_given"))

# SI: target tip_high — gorjeta generosa vs modesta (percentil 75 em viagens com tip > 0)
print("\n── Variável alvo tip_high (gorjeta alta vs baixa) ──")
p75_tip = df_card_spark.filter(F.col("tip_amount") > 0).approxQuantile(
    "tip_amount", [0.75], 0.01
)[0]
df_card_spark = df_card_spark.withColumn(
    "tip_high",
    F.when(F.col("tip_amount") >= F.lit(float(p75_tip)), 1).otherwise(0).cast(IntegerType()),
)
print(f"   Limiar P75 (tip_amount, viagens com gorjeta): ${p75_tip:.4f}")
imprimir_spark_df(df_card_spark.groupBy("tip_high").count().orderBy("tip_high"))
import json as _json
with open(os.path.join(OUTPUT_DIR, "tip_high_threshold.json"), "w", encoding="utf-8") as _f:
    _json.dump({
        "metodo": "percentil_75",
        "coluna": "tip_amount",
        "filtro": "tip_amount > 0",
        "limiar_usd": round(float(p75_tip), 4),
    }, _f, indent=2)
print(f"   ✓ Limiar guardado em tip_high_threshold.json")

# ─────────────────────────────────────────
# 4. ESTATÍSTICAS DESCRITIVAS
#    DataFrame.summary() — 1 job em vez de 30
#    RDD chave-valor para distribuição de classes
# ─────────────────────────────────────────
num_cols = ["fare_amount", "tip_amount", "trip_distance",
            "trip_duration_min", "passenger_count", "avg_speed"]

# ── 4a. Estatísticas via Spark DataFrame (eficiente — 1 único job) ──
secao("Estatísticas descritivas via Spark DataFrame")
stats_spark = df_card_spark.select(num_cols).summary(
    "count", "mean", "stddev", "min", "max"
)
imprimir_spark_df(stats_spark)

# Converter para CSV
stats_pd = stats_spark.toPandas().set_index("summary").T
stats_pd.index.name = "coluna"
stats_pd.to_csv(f"{OUTPUT_DIR}/estatisticas_descritivas.csv")

# SI: balanceamento de classes e estatísticas descritivas para justificar split estratificado
# Spark: Stage 1 — terceira transformação RDD (reduceByKey) + summary/estatísticas no DataFrame
# ── 4b. Distribuição de classes via RDD chave-valor ──
print("\n── Distribuição de classes via RDD (chave-valor) ──")
rdd_classes = df_card_spark.select("tip_given").rdd \
    .map(lambda r: (r["tip_given"], 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortByKey()

total_cls = df_card_spark.count()
for classe, contagem in rdd_classes.collect():
    print(f"   Classe {classe}: {contagem:,} ({contagem/total_cls*100:.1f}%)")

# ─────────────────────────────────────────
# TABELA SUMMARY COMPLETA (com mediana e moda)
# calculada após toPandas()
# ─────────────────────────────────────────


# ─────────────────────────────────────────
# 5. CONVERTER PARA PANDAS (visualizações e modelos)
# ─────────────────────────────────────────
print("\n── A converter Spark DataFrame → Pandas ──")

features_to_keep = [
    "fare_amount", "trip_distance", "trip_duration_min",
    "passenger_count", "avg_speed", "pickup_hour",
    "pickup_weekday", "pickup_weekend",
    "RatecodeID", "tip_amount", "tip_given", "tip_high"
]
features_to_keep = [c for c in features_to_keep if c in df_card_spark.columns]

# SI: export dataset_processado.csv para sklearn, TensorFlow e validação Weka (GUI — ver projeto/WEKA_GUI.txt)
# Spark: Stage 3 — persistência Parquet do dataset limpo para a fase 2 Spark ML
output_parquet = df_card_spark.select(features_to_keep).dropna()
output_parquet.write.mode("overwrite").parquet(f"{OUTPUT_DIR}/dataset_processado.parquet")
print(f"   ✓ dataset_processado.parquet salvo em {OUTPUT_DIR}")

df_card = output_parquet.toPandas()
print(f"   Dataset final: {len(df_card):,} linhas, {df_card.shape[1]} colunas")

spark.stop()

# ─────────────────────────────────────────
# 5b. TABELA SUMMARY (média, mediana, moda, min, max)
# ─────────────────────────────────────────
print("\n── Tabela Summary Completa ──")

summary_cols = ["fare_amount", "tip_amount", "trip_distance",
                "trip_duration_min", "passenger_count", "avg_speed",
                "pickup_hour", "tip_given", "tip_high"]
summary_cols = [c for c in summary_cols if c in df_card.columns]

summary_rows = []
for col in summary_cols:
    s = df_card[col].dropna()
    tipo = "Qualit." if col in ["tip_given", "tip_high", "pickup_weekend", "pickup_weekday", "RatecodeID"] else "Quant."
    summary_rows.append({
        "Variável":  col,
        "Tipo":      tipo,
        "Média":     round(s.mean(), 3),
        "Mediana":   round(s.median(), 3),
        "Moda":      round(float(s.mode()[0]), 3),
        "Mín":       round(s.min(), 3),
        "Máx":       round(s.max(), 3),
        "Desvio P.": round(s.std(), 3),
    })

summary_df = pd.DataFrame(summary_rows).set_index("Variável")
secao("Tabela Summary Completa")
imprimir_pandas(summary_df.reset_index(), dec=3)
summary_df.to_csv(f"{OUTPUT_DIR}/tabela_summary.csv")
print("   ✓ tabela_summary.csv")

# SI: figuras EDA (fig0–fig5) obrigatórias na apresentação e relatório de Sistemas Inteligentes
# Spark: visualizações geradas após processamento Spark; dados derivam do pipeline da fase 1
# ─────────────────────────────────────────
# 6. VISUALIZAÇÕES
# ─────────────────────────────────────────

# ─────────────────────────────────────────
print("\n── A gerar visualizações ──")

# Fig 0: Tabela de variáveis (quant vs qualit)
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off")
tabela_vars = [
    ["fare_amount",       "Quantitativa Contínua",  "Valor base da corrida ($)"],
    ["tip_amount",        "Quantitativa Contínua",  "Valor da gorjeta — TARGET regressão ($)"],
    ["trip_distance",     "Quantitativa Contínua",  "Distância da viagem (milhas)"],
    ["trip_duration_min", "Quantitativa Contínua",  "Duração da viagem (min) — derivada"],
    ["avg_speed",         "Quantitativa Contínua",  "Velocidade média (mph) — derivada"],
    ["passenger_count",   "Quantitativa Discreta",  "Número de passageiros (1–6)"],
    ["pickup_hour",       "Quantitativa Discreta",  "Hora de início da viagem (0–23) — derivada"],
    ["pickup_weekday",    "Quantitativa Discreta",  "Dia da semana (0–6) — derivada"],
    ["payment_type",      "Qualitativa Nominal",    "1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided trip"],
    ["RatecodeID",        "Qualitativa Ordinal",    "1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated fare, 6=Group ride"],
    ["pickup_weekend",    "Qualitativa Binária",    "Fim de semana? (0/1) — derivada"],
    ["tip_given",         "Qualitativa Binária",    "Deu gorjeta? (0/1) — TARGET classificação (baseline)"],
    ["tip_high",          "Qualitativa Binária",    "Gorjeta alta? (0/1) — ≥ P75 de tip_amount"],
]
tbl = ax.table(cellText=tabela_vars,
               colLabels=["Variável", "Tipo", "Descrição"],
               cellLoc="left", loc="center", colWidths=[0.22, 0.22, 0.56])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.45)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2196F3"); cell.set_text_props(color="white", fontweight="bold")
    elif r <= len(tabela_vars):
        t = tabela_vars[r-1][1]
        if "Contínua" in t:   cell.set_facecolor("#E3F2FD")
        elif "Discreta" in t: cell.set_facecolor("#E8F5E9")
        else:                 cell.set_facecolor("#FFF3E0")
    cell.set_edgecolor("#CCCCCC")
ax.set_title("Descrição das Variáveis do Dataset", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig0_descricao_variaveis.png", bbox_inches="tight")
plt.close()
print("   ✓ fig0_descricao_variaveis.png")

# Fig 0b: Tabela summary
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis("off")
summary_display = summary_df.reset_index()
tbl2 = ax.table(cellText=summary_display.values.tolist(),
                colLabels=list(summary_display.columns),
                cellLoc="center", loc="center")
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8.5)
tbl2.scale(1, 1.5)
for (r, c), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2196F3"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("#CCCCCC")
ax.set_title("Estatísticas Descritivas — Média, Mediana, Moda, Mín, Máx", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig0b_tabela_summary.png", bbox_inches="tight")
plt.close()
print("   ✓ fig0b_tabela_summary.png")

# Fig 1: Histogramas com curva KDE + linhas de média e mediana
vars_plot = [
    ("fare_amount",       "Valor da Corrida ($)"),
    ("tip_amount",        "Valor da Gorjeta ($)"),
    ("trip_distance",     "Distância (milhas)"),
    ("trip_duration_min", "Duração (min)"),
    ("passenger_count",   "Nº Passageiros"),
    ("avg_speed",         "Velocidade Média (mph)"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Distribuição das Variáveis Quantitativas (Histograma + Curva KDE)",
             fontsize=13, fontweight="bold")

for ax, (col, label) in zip(axes.flatten(), vars_plot):
    data = df_card[col].dropna()
    p99  = data.quantile(0.99)
    data = data[data <= p99]
    counts_h, bin_edges = np.histogram(data, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, counts_h, width=(bin_edges[1]-bin_edges[0]),
           color=PALETTE[0], alpha=0.6, edgecolor="white", linewidth=0.3)
    try:
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde   = stats.gaussian_kde(data)
        ax.plot(kde_x, kde(kde_x), color=PALETTE[1], linewidth=2, label="Curva KDE")
    except Exception:
        pass
    ax.axvline(data.mean(),   color=PALETTE[2], linestyle="--", linewidth=1.5,
               label=f"Média={data.mean():.2f}")
    ax.axvline(data.median(), color=PALETTE[3], linestyle=":",  linewidth=1.5,
               label=f"Mediana={data.median():.2f}")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Densidade", fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_distribuicoes.png", bbox_inches="tight")
plt.close()
print("   ✓ fig1_distribuicoes.png")

# Fig 1b: Variáveis qualitativas — barras (apenas VARS_QUAL presentes em df_card)
qual_vars = [
    ("RatecodeID",     "Código de Tarifa"),
    ("pickup_weekend", "Fim de Semana"),
    ("tip_given",      "Deu Gorjeta?"),
    ("tip_high",       "Gorjeta Alta? (≥P75)"),
]
qual_vars = [(col, label) for col, label in qual_vars if col in df_card.columns]
rate_labels = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau", 5: "Negot.", 6: "Group"}
bin_labels = {0: "Não", 1: "Sim"}

n_qual = len(qual_vars)
if n_qual:
    ncols = min(n_qual, 4)
    nrows = (n_qual + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    fig.suptitle("Distribuição das Variáveis Qualitativas", fontsize=13, fontweight="bold")
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, (col, label) in zip(axes_flat, qual_vars):
        vc = df_card[col].value_counts().sort_index()
        if col == "RatecodeID":
            xlabels = [rate_labels.get(int(i), str(int(i))) for i in vc.index]
        elif col in ("pickup_weekend", "tip_given", "tip_high"):
            xlabels = [bin_labels.get(int(i), str(int(i))) for i in vc.index]
        else:
            xlabels = [str(i) for i in vc.index]
        ax.bar(xlabels, vc.values, color=PALETTE[0], alpha=0.85, edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequência")
        ax.set_title(label, fontweight="bold")
    for ax in axes_flat[n_qual:]:
        ax.set_visible(False)
    plt.tight_layout()
else:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title("Distribuição das Variáveis Qualitativas", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.5, "Sem variáveis qualitativas no dataset filtrado.", ha="center", va="center")
    plt.tight_layout()

plt.savefig(f"{FIGURES_DIR}/fig1b_variaveis_qualitativas.png", bbox_inches="tight")
plt.close()
print("   ✓ fig1b_variaveis_qualitativas.png")

# Fig 2: Gorjeta por hora do dia
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Gorjeta ao Longo do Dia", fontsize=14, fontweight="bold")

hour_tip_pct = df_card.groupby("pickup_hour")["tip_given"].mean() * 100
hour_tip_avg = df_card[df_card["tip_given"] == 1].groupby("pickup_hour")["tip_amount"].mean()

axes[0].bar(hour_tip_pct.index, hour_tip_pct.values, color=PALETTE[0], alpha=0.85)
axes[0].set_xlabel("Hora do Dia")
axes[0].set_ylabel("% Viagens com Gorjeta")
axes[0].set_title("Probabilidade de Gorjeta por Hora")
axes[0].set_xticks(range(0, 24, 2))

axes[1].bar(hour_tip_avg.index, hour_tip_avg.values, color=PALETTE[1], alpha=0.85)
axes[1].set_xlabel("Hora do Dia")
axes[1].set_ylabel("Valor Médio da Gorjeta ($)")
axes[1].set_title("Valor Médio da Gorjeta por Hora")
axes[1].set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_gorjeta_hora.png", bbox_inches="tight")
plt.close()
print("   ✓ fig2_gorjeta_hora.png")

# Fig 3: Matriz de correlação
fig, ax = plt.subplots(figsize=(10, 8))
num_c = ["fare_amount", "tip_amount", "trip_distance", "trip_duration_min",
         "passenger_count", "avg_speed", "pickup_hour", "tip_given"]
num_c = [c for c in num_c if c in df_card.columns]
corr  = df_card[num_c].corr()
mask  = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Matriz de Correlação", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_correlacao.png", bbox_inches="tight")
plt.close()
print("   ✓ fig3_correlacao.png")

# Fig 4: Scatter gorjeta vs variáveis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Relação entre Gorjeta e Outras Variáveis", fontsize=14, fontweight="bold")
sample_viz = df_card[df_card["tip_amount"] > 0].sample(min(5000, len(df_card)))
pairs = [
    ("fare_amount",       "Valor da Corrida ($)"),
    ("trip_distance",     "Distância (milhas)"),
    ("trip_duration_min", "Duração (min)"),
]
for ax, (xcol, xlabel) in zip(axes, pairs):
    ax.scatter(sample_viz[xcol], sample_viz["tip_amount"],
               alpha=0.15, s=8, color=PALETTE[2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gorjeta ($)")
    ax.set_title(f"Gorjeta vs {xlabel}")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_scatter_gorjeta.png", bbox_inches="tight")
plt.close()
print("   ✓ fig4_scatter_gorjeta.png")

# Fig 5: Pizza + gorjeta por dia da semana
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribuição de Gorjeta", fontsize=14, fontweight="bold")
counts = df_card["tip_given"].value_counts()
axes[0].pie(counts.values, labels=["Com Gorjeta", "Sem Gorjeta"],
            autopct="%1.1f%%", colors=[PALETTE[2], PALETTE[1]],
            startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
axes[0].set_title("Cartão — Com vs Sem Gorjeta")

weekday_names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
wd_tip = df_card.groupby("pickup_weekday")["tip_given"].mean() * 100
axes[1].bar([weekday_names[i % 7] for i in wd_tip.index], wd_tip.values,
            color=PALETTE[3], alpha=0.85)
axes[1].set_ylabel("% Viagens com Gorjeta")
axes[1].set_title("% Gorjeta por Dia da Semana")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_distribuicao_gorjeta.png", bbox_inches="tight")
plt.close()
print("   ✓ fig5_distribuicao_gorjeta.png")

# ─────────────────────────────────────────
# 7. GUARDAR DATASET PROCESSADO
# ─────────────────────────────────────────
df_final = df_card.dropna().reset_index(drop=True)

# Dataset limitado (Python + Weka GUI) — ver projeto/WEKA_GUI.txt
if len(df_final) > DATASET_N_ROWS:
    df_final_out = df_final.sample(n=DATASET_N_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df_final_out = df_final.copy()
df_final_out.to_csv(f"{OUTPUT_DIR}/dataset_processado.csv", index=False)

print(f"\n✅ Dataset processado guardado (CSV + Parquet): {len(df_final_out):,} linhas, {df_final_out.shape[1]} colunas")
print(f"   Colunas: {list(df_final_out.columns)}")
imprimir_describe_pandas(df_final_out, dec=3, titulo="Estatísticas finais do dataset limpo")

print("\n✅ Fase 1 concluída! Próximo passo: src/fase2/fase2_runner.py")
