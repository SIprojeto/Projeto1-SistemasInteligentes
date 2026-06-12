"""
Configurações partilhadas — caminhos e sessão Spark.
Importado por todos os módulos da fase 2.
"""

import logging
import os
import sys

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYTHONWARNINGS"]        = "ignore"
# Reduz ruído dos workers Python em modo local (BrokenPipeError no daemon é benigno)
os.environ.setdefault("PYSPARK_PIN_THREAD", "1")

logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, IntegerType
)

# ── Caminhos ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "fase2_ml")   # ← NOVO: pasta dedicada Fase 2
MODELS_DIR  = os.path.join(BASE_DIR, "outputs", "models")
SPLITS_DIR  = os.path.join(BASE_DIR, "outputs", "splits")

for d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42

# SI: grelha completa + 3 folds (alinhado com projeto/bigDataSpark.txt — Stage 7)
# Spark: FAST_TUNING=1 opcional só para teste rápido manual (não usar na entrega)
FAST_TUNING = os.environ.get("FAST_TUNING", "").lower() in ("1", "true", "yes")

# Flags globais para controlar artefactos em disco.
# - WRITE_REPORTS=0: não escrever CSV/TXT/JSON em outputs/reports (mantém prints/figuras).
# - WRITE_MODELS=0: não escrever modelos/pipeline em outputs/models.
# - WRITE_SPLITS=0: não persistir splits em outputs/splits (a execução mantém-se em memória).
WRITE_REPORTS = os.environ.get("WRITE_REPORTS", "1").lower() not in ("0", "false", "no")
WRITE_MODELS = os.environ.get("WRITE_MODELS", "1").lower() not in ("0", "false", "no")
WRITE_SPLITS = os.environ.get("WRITE_SPLITS", "1").lower() not in ("0", "false", "no")


def should_write_reports() -> bool:
    return WRITE_REPORTS


def should_write_models() -> bool:
    return WRITE_MODELS


def should_write_splits() -> bool:
    return WRITE_SPLITS


def cv_num_folds():
    return 2 if FAST_TUNING else 3


def cv_parallelism():
    # parallelism>1 usa multiprocessing.Pool no PySpark ML tuning; em local[*]
    # (Docker/Windows) provoca TypeError: 'SparkSession' object is not callable
    return 1

# ── Schema explícito do dataset processado ────────────────────
DATASET_SCHEMA = StructType([
    StructField("fare_amount",        DoubleType(),  True),
    StructField("trip_distance",      DoubleType(),  True),
    StructField("trip_duration_min",  DoubleType(),  True),
    StructField("passenger_count",    IntegerType(), True),
    StructField("avg_speed",          DoubleType(),  True),
    StructField("pickup_hour",        IntegerType(), True),
    StructField("pickup_weekday",     IntegerType(), True),
    StructField("pickup_weekend",     IntegerType(), True),
    StructField("RatecodeID",         IntegerType(), True),
    StructField("tip_amount",         DoubleType(),  True),
    StructField("tip_given",          IntegerType(), True),
    StructField("tip_high",           IntegerType(), True),
])

FEATURE_COLS = [
    "fare_amount", "trip_distance", "trip_duration_min",
    "passenger_count", "avg_speed", "pickup_hour",
    "pickup_weekday", "pickup_weekend", "RatecodeID"
]


def criar_spark(app_name="NYC_Taxi_ML"):
    """Cria e devolve uma SparkSession configurada."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# SI: infraestrutura Spark partilhada; CSV no Windows, Parquet em Docker/Linux
# Spark: Stage 3 — leitura com StructType explícito, printSchema() e describe().show()
def carregar_dataset(spark):
    """Carrega o dataset processado com schema explícito."""
    parquet_path = os.path.join(OUTPUT_DIR, "dataset_processado.parquet")
    csv_path     = os.path.join(OUTPUT_DIR, "dataset_processado.csv")

    # CSV preferido no Windows (Parquet local requer winutils/hadoop nativo)
    use_csv_first = sys.platform == "win32" or not os.path.exists(parquet_path)

    if not use_csv_first and os.path.exists(parquet_path):
        try:
            print(f"   Carregando de Parquet (schema explícito): {parquet_path}")
            df = spark.read.schema(DATASET_SCHEMA).parquet(parquet_path)
        except Exception as exc:
            print(f"   ⚠ Falha ao ler Parquet ({exc.__class__.__name__}); a usar CSV")
            use_csv_first = True

    if use_csv_first:
        print(f"   Carregando de CSV (schema explícito): {csv_path}")
        df = spark.read.schema(DATASET_SCHEMA).option("header", True).csv(csv_path)

    if "tip_high" not in df.columns:
        from pyspark.sql import functions as F
        p75 = df.filter(F.col("tip_amount") > 0).approxQuantile("tip_amount", [0.75], 0.01)[0]
        df = df.withColumn(
            "tip_high",
            F.when(F.col("tip_amount") >= F.lit(float(p75)), 1).otherwise(0).cast("int"),
        )
        print(f"   ℹ Coluna tip_high ausente — recalculada (P75 tip_amount = ${p75:.4f})")

    n = df.count()
    print(f"   Linhas carregadas para ML (fase 2+): {n:,}")
    print(f"   Colunas: {len(df.columns)}")
    if "tip_high" in df.columns:
        bal = df.groupBy("tip_high").count().orderBy("tip_high").collect()
        parts = ", ".join(f"classe {r['tip_high']}: {r['count']:,}" for r in bal)
        print(f"   Balanceamento tip_high: {parts}")
    print("\n── Schema do dataset ──")
    df.printSchema()
    _src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from spark_display import imprimir_describe_spark
    imprimir_describe_spark(df, titulo="── Estatísticas descritivas ──")
    return df
