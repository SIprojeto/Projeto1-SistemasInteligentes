"""
Spark ML Pipeline — pré-processamento e splits.

Responsabilidades:
  - Split estratificado train/val/test (70/15/15)
  - Pipeline: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler
  - Fit apenas em treino; transform em val e test (evita data leakage)
  - Persistência dos splits em Parquet e CSV
"""

import os
import sys

_src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)
from spark_display import imprimir_amostra_spark

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

from .spark_setup import RANDOM_STATE, FEATURE_COLS, MODELS_DIR, SPLITS_DIR, OUTPUT_DIR, should_write_splits, should_write_models


def _persistir_split(df, nome):
    """Persiste split em Parquet (Linux/Docker); fallback CSV no Windows sem winutils."""
    if not should_write_splits():
        return
    parquet_path = os.path.join(SPLITS_DIR, f"{nome}.parquet")
    csv_path = os.path.join(SPLITS_DIR, f"{nome}.csv")
    try:
        df.write.mode("overwrite").parquet(parquet_path)
    except Exception as exc:
        print(f"   ⚠ Parquet indisponível para {nome} ({exc.__class__.__name__}); a usar CSV")
        df.toPandas().to_csv(csv_path, index=False)


# SI: divisão dos dados para treino de SVM, J48, MLP, Naive Bayes e comparação com DL
# Spark: Stage 5 — split train/validation/test estratificado, seed fixo, Parquet em outputs/splits/
def fazer_splits(df, label_col="tip_given", split_prefix=""):
    """Split estratificado por classe (70/15/15) com seed fixo."""
    titulo = "tip_high (gorjeta alta vs baixa)" if label_col == "tip_high" else "tip_given (com vs sem gorjeta)"
    print(f"\n── Split estratificado — alvo: {titulo} (70/15/15) ──")
    print("   Justificação: 70% treino garante dados suficientes para tuning.")
    print("   15% validação para TrainValidationSplit (hiperparâmetros). 15% teste para avaliação final.")
    print(f"   Seed fixo: {RANDOM_STATE} (reprodutibilidade total)")

    df_zero = df.filter(F.col(label_col) == 0)
    df_one  = df.filter(F.col(label_col) == 1)

    tr0, va0, te0 = df_zero.randomSplit([0.7, 0.15, 0.15], seed=RANDOM_STATE)
    tr1, va1, te1 = df_one.randomSplit([0.7, 0.15, 0.15], seed=RANDOM_STATE)

    train = tr0.union(tr1).cache()
    val   = va0.union(va1).cache()
    test  = te0.union(te1).cache()

    print(f"\n   Train : {train.count():,} linhas")
    print(f"   Val   : {val.count():,} linhas")
    print(f"   Test  : {test.count():,} linhas")
    print(f"   Classe 0 treino : {tr0.count():,}")
    print(f"   Classe 1 treino : {tr1.count():,}")

    prefix = split_prefix or ""
    _persistir_split(train, f"{prefix}train" if prefix else "train")
    _persistir_split(val, f"{prefix}validation" if prefix else "validation")
    _persistir_split(test, f"{prefix}test" if prefix else "test")
    if should_write_splits():
        print("   ✓ Splits persistidos em Parquet/CSV (outputs/splits/)")
    else:
        print("   ℹ WRITE_SPLITS=0: splits não persistidos em disco")

    return train, val, test


NUMERIC_FEATURE_COLS = [c for c in FEATURE_COLS if c != "RatecodeID"]


# SI: features normalizadas partilhadas entre Spark ML e sklearn (via conversão NumPy)
# Spark: Stage 6 — Pipeline com StringIndexer + OneHotEncoder + VectorAssembler + StandardScaler
def construir_pipeline(train, val, test):
    """
    Constrói e treina o Pipeline Spark ML.
    Fit apenas em train → transform em val e test.
    RatecodeID: StringIndexer + OneHotEncoder (categórica nominal, não ordinal).
    """
    print("\n── Construindo Spark ML Pipeline ──")
    print("   Etapas: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler")

    train = train.withColumn("RatecodeID_str", F.col("RatecodeID").cast("string"))
    val   = val.withColumn("RatecodeID_str", F.col("RatecodeID").cast("string"))
    test  = test.withColumn("RatecodeID_str", F.col("RatecodeID").cast("string"))

    indexer = StringIndexer(
        inputCol="RatecodeID_str", outputCol="RatecodeID_idx", handleInvalid="keep"
    )
    ohe = OneHotEncoder(
        inputCol="RatecodeID_idx", outputCol="RatecodeID_ohe", dropLast=True
    )
    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURE_COLS + ["RatecodeID_ohe"],
        outputCol="raw_features",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withMean=True, withStd=True
    )

    pipeline = Pipeline(stages=[indexer, ohe, assembler, scaler])

    print("   A treinar pipeline (fit apenas em train)...")
    pipeline_model = pipeline.fit(train)
    if should_write_models():
        try:
            pipeline_model.write().overwrite().save(os.path.join(MODELS_DIR, "spark_pipeline"))
            print("   ✓ Pipeline guardada em outputs/models/spark_pipeline")
        except Exception as exc:
            print(f"   ⚠ Não foi possível guardar pipeline em disco ({exc.__class__.__name__})")
            print("     (normal no Windows sem winutils; usar Docker/Linux para persistência Parquet)")
    else:
        print("   ℹ WRITE_MODELS=0: pipeline não guardada em disco")

    train_prep = pipeline_model.transform(train).cache()
    val_prep   = pipeline_model.transform(val).cache()
    test_prep  = pipeline_model.transform(test).cache()

    print("\n── Exemplo de features pré-processadas ──")
    imprimir_amostra_spark(
        train_prep, n=3, cols=["features", "tip_given", "tip_amount"], cols_vetor=("features",)
    )

    return train_prep, val_prep, test_prep, pipeline_model
