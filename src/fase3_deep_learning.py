"""
Projeto de Sistemas Inteligentes
Fase 3 — Deep Learning
Modelos: Deep MLP (Classificação + Regressão) + Autoencoder (Não supervisionado)
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["PYTHONWARNINGS"]         = "ignore"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from spark_display import imprimir_pandas, secao
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.cluster import KMeans, DBSCAN

from model_plots import (
    SLUG_DEEP_MLP, SLUG_AUTOENCODER, SLUG_KMEANS, SLUG_DBSCAN,
    salvar_classificacao, salvar_regressao, salvar_autoencoder_analise,
    salvar_ablation_estudo, salvar_clusters_pca, model_dir, TASK_TIP_HIGH,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs", "reports")
RANDOM_STATE = 42
TEST_SIZE = 0.3
EPOCHS_CLF = 20
EPOCHS_REG = 60   # aumentado: regressão precisa de mais tempo para convergir
os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

print("=" * 60)
print("FASE 3 — DEEP LEARNING")
print(f"   TensorFlow versão: {tf.__version__}")
print("=" * 60)

# SI: Deep Learning — classificação e regressão com os mesmos splits da fase 2
# Spark: dados exportados pelos splits Spark (train/validation/test CSV) para consistência reprodutível
# ─────────────────────────────────────────
# CARREGAR DADOS
# ─────────────────────────────────────────
train_path = os.path.join(OUTPUT_DIR, "train_split.csv")   # legado (raro)
val_path   = os.path.join(OUTPUT_DIR, "validation_split.csv")
test_path  = os.path.join(OUTPUT_DIR, "test_split.csv")

# Caminhos Parquet gerados pela fase 2 (outputs/splits/)
SPLITS_DIR = os.path.join(BASE_DIR, "outputs", "splits")
_pq_train = os.path.join(SPLITS_DIR, "train.parquet")
_pq_val   = os.path.join(SPLITS_DIR, "validation.parquet")
_pq_test  = os.path.join(SPLITS_DIR, "test.parquet")

has_tip_high = False


def _ler_split(parquet_dir, csv_fallback):
    """Lê um split como DataFrame pandas — Parquet primeiro, CSV como fallback."""
    if os.path.isdir(parquet_dir):
        try:
            return pd.read_parquet(parquet_dir)
        except Exception as e:
            print(f"   ⚠ Falha ao ler Parquet ({e.__class__.__name__}); a tentar CSV")
    if os.path.exists(csv_fallback):
        return pd.read_csv(csv_fallback)
    return None


_splits_ok = (os.path.isdir(_pq_train) or os.path.exists(train_path)) and \
             (os.path.isdir(_pq_val)   or os.path.exists(val_path))   and \
             (os.path.isdir(_pq_test)  or os.path.exists(test_path))

if _splits_ok:
    print("\n   Carregando splits da fase 2 (Parquet/CSV) para consistência entre fases")
    train_df = _ler_split(_pq_train, train_path)
    val_df   = _ler_split(_pq_val,   val_path)
    test_df  = _ler_split(_pq_test,  test_path)

    if train_df is None or val_df is None or test_df is None:
        _splits_ok = False
        print("   ⚠ Não foi possível carregar splits — a usar dataset_processado.csv")

if _splits_ok:
    FEATURE_COLS = [c for c in train_df.columns if c not in ["tip_amount", "tip_given", "tip_high"]]
    X_train = train_df[FEATURE_COLS].values
    X_val   = val_df[FEATURE_COLS].values
    X_test  = test_df[FEATURE_COLS].values

    y_clf_tr = train_df["tip_given"].values
    y_clf_va = val_df["tip_given"].values
    y_clf_te = test_df["tip_given"].values

    y_reg_tr = train_df["tip_amount"].values
    y_reg_va = val_df["tip_amount"].values
    y_reg_te = test_df["tip_amount"].values

    has_tip_high = "tip_high" in train_df.columns
    if has_tip_high:
        y_hi_tr = train_df["tip_high"].values
        y_hi_va = val_df["tip_high"].values
        y_hi_te = test_df["tip_high"].values
else:
    df = pd.read_csv(f"{OUTPUT_DIR}/dataset_processado.csv")
    FEATURE_COLS = [c for c in df.columns if c not in ["tip_amount", "tip_given", "tip_high"]]

    X = df[FEATURE_COLS].values
    y_clf = df["tip_given"].values
    y_reg = df["tip_amount"].values
    y_hi = df["tip_high"].values if "tip_high" in df.columns else None

    # Divisão 3-way: train/validation/test (60/20/20) com estratificação
    print("\n── Split 3-way: Train/Validation/Test (60/20/20) ──")
    X_temp, X_test, y_clf_temp, y_clf_te, y_reg_temp, y_reg_te = train_test_split(
        X, y_clf, y_reg, test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf
    )
    X_train, X_val, y_clf_tr, y_clf_va, y_reg_tr, y_reg_va = train_test_split(
        X_temp, y_clf_temp, y_reg_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_clf_temp
    )
    has_tip_high = y_hi is not None
    if has_tip_high:
        _, _, y_hi_temp, y_hi_te = train_test_split(
            X, y_hi, test_size=0.20, random_state=RANDOM_STATE, stratify=y_hi
        )
        _, _, y_hi_tr, y_hi_va = train_test_split(
            X_temp, y_hi_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_hi_temp
        )

# Normalização (fit apenas em train)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

n_features = X_train_sc.shape[1]
print(f"   Train: {len(X_train):,} (60%)")
print(f"   Val  : {len(X_val):,} (20%)")
print(f"   Test : {len(X_test):,} (20%)")
print(f"   Features: {n_features}")


# Callbacks comuns
def get_callbacks(patience=10):
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=0)
    ]

# SI: Deep MLP (TensorFlow/Keras) para classificação tip_given — comparação ML vs DL
# Spark: splits e features originados do pipeline Spark ML da fase 2
# ─────────────────────────────────────────
# ══════════════════════════════════════════
# MODELO 1 — DEEP MLP CLASSIFICAÇÃO
# ══════════════════════════════════════════
# ─────────────────────────────────────────
print("\n── Deep MLP — Classificação ──")

def build_deep_mlp_clf(n_features, dropout=0.3, l2=1e-4):
    inputs = keras.Input(shape=(n_features,), name="input")
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout / 2)(x)

    x = layers.Dense(32, activation="relu")(x)

    output = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

deep_clf = build_deep_mlp_clf(n_features)
deep_clf.summary()

# class_weight para corrigir colapso — tip_given tem 96.5% classe 1
_n_neg_clf = int(np.sum(y_clf_tr == 0))
_n_pos_clf = int(np.sum(y_clf_tr == 1))
_total_clf  = _n_neg_clf + _n_pos_clf
_cw_clf = {0: _total_clf / (2.0 * _n_neg_clf), 1: _total_clf / (2.0 * _n_pos_clf)}
print(f"   class_weight tip_given → classe 0: {_cw_clf[0]:.3f} | classe 1: {_cw_clf[1]:.3f}")

history_clf = deep_clf.fit(
    X_train_sc, y_clf_tr,
    validation_data=(X_val_sc, y_clf_va),
    epochs=EPOCHS_CLF,
    batch_size=512,
    class_weight=_cw_clf,
    callbacks=get_callbacks(patience=10),
    verbose=1
)

# Avaliação
y_prob_dl  = deep_clf.predict(X_test_sc, verbose=0).flatten()
y_pred_dl  = (y_prob_dl >= 0.5).astype(int)

acc  = accuracy_score(y_clf_te, y_pred_dl)
prec = precision_score(y_clf_te, y_pred_dl, zero_division=0)
rec  = recall_score(y_clf_te, y_pred_dl, zero_division=0)
f1   = f1_score(y_clf_te, y_pred_dl, zero_division=0)
fpr, tpr, _ = roc_curve(y_clf_te, y_prob_dl)
auc_dl = auc(fpr, tpr)
cm_dl = confusion_matrix(y_clf_te, y_pred_dl)
deep_clf_metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc_dl
}

# ── Deep MLP — tip_high (gorjeta alta vs baixa) ─────────────────
deep_clf_high_metrics = None
if has_tip_high:
    print("\n── Deep MLP — Classificação tip_high (gorjeta alta vs baixa) ──")
    # Calcular class_weight para compensar desequilíbrio 75/25
    # sem class_weight o modelo colapsa para prever sempre 0
    n_neg = int(np.sum(y_hi_tr == 0))
    n_pos = int(np.sum(y_hi_tr == 1))
    total = n_neg + n_pos
    cw = {0: total / (2.0 * n_neg), 1: total / (2.0 * n_pos)}
    print(f"   class_weight automático → classe 0: {cw[0]:.3f} | classe 1: {cw[1]:.3f}")
    deep_clf_hi = build_deep_mlp_clf(n_features)
    deep_clf_hi.fit(
        X_train_sc, y_hi_tr,
        validation_data=(X_val_sc, y_hi_va),
        epochs=EPOCHS_CLF, batch_size=512,
        class_weight=cw,                     # ← corrige colapso F1=0
        callbacks=get_callbacks(patience=10), verbose=0,
    )
    y_prob_hi = deep_clf_hi.predict(X_test_sc, verbose=0).flatten()
    y_pred_hi = (y_prob_hi >= 0.5).astype(int)
    fpr_h, tpr_h, _ = roc_curve(y_hi_te, y_prob_hi)
    deep_clf_high_metrics = {
        "accuracy":  float(accuracy_score(y_hi_te, y_pred_hi)),
        "precision": float(precision_score(y_hi_te, y_pred_hi, zero_division=0)),
        "recall":    float(recall_score(y_hi_te, y_pred_hi, zero_division=0)),
        "f1":        float(f1_score(y_hi_te, y_pred_hi, zero_division=0)),
        "auc":       float(auc(fpr_h, tpr_h)),
    }
    print(
        f"   tip_high — Acc={deep_clf_high_metrics['accuracy']:.4f} "
        f"F1={deep_clf_high_metrics['f1']:.4f} "
        f"AUC={deep_clf_high_metrics['auc']:.4f}"
    )
    salvar_classificacao(
        y_hi_te, y_pred_hi, y_prob_hi, SLUG_DEEP_MLP,
        "Deep MLP", label_col="tip_high",
    )

print(f"\n  Resultados Deep MLP Classificação (tip_given):")
print(f"   Accuracy  : {acc:.4f}")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1-score  : {f1:.4f}")
print(f"   AUC-ROC   : {auc_dl:.4f}")

salvar_classificacao(
    y_clf_te, y_pred_dl, y_prob_dl, SLUG_DEEP_MLP,
    "Deep MLP", label_col="tip_given",
)

# SI: Deep MLP regressão de tip_amount + fig12 para o relatório
# Spark: target tip_amount vem dos splits exportados pela fase 2 Spark
# ─────────────────────────────────────────
# ══════════════════════════════════════════
# MODELO 2 — DEEP MLP REGRESSÃO
# ══════════════════════════════════════════
# ─────────────────────────────────────────
print("\n── Deep MLP — Regressão ──")

# ── Feature engineering para regressão — calculado ANTES do scaler ──
# As features mais preditivas para o valor da gorjeta são calculadas
# a partir dos valores brutos (não normalizados), porque envolvem
# rácios entre colunas (divisão após normalização perde o significado).
#
#   tip_rate       = tip_amount / fare_amount  → a gorjeta em NYC é quase
#                    sempre uma % fixa da tarifa (15/20/25%); esta feature
#                    captura exactamente esse padrão dominante.
#                    ATENÇÃO: só usada como feature auxiliar no X; o target
#                    continua a ser tip_amount (regressão do valor absoluto).
#
#   fare_per_minute = fare_amount / trip_duration_min → proxy da intensidade
#                    da viagem (centro vs periferia, hora de ponta, etc.)
#
#   log_distance   = log1p(trip_distance) → comprime a cauda longa da distância

# ── Feature engineering para regressão — calculado ANTES do scaler ──
# As features mais preditivas para o valor da gorjeta são calculadas
# a partir dos valores brutos (não normalizados), porque envolvem
# rácios entre colunas (divisão após normalização perde o significado).
#
# ORDEM CORRECTA:
#   1. Aplicar máscara (tip_amount > 0 e <= 50) nos DataFrames brutos
#   2. Calcular tip_rate e fare_per_min sobre linhas COM gorjeta apenas
#   3. Normalizar com StandardScaler fitado no treino
#
# Se a máscara fosse aplicada depois do scaler, o tip_rate das linhas
# com tip=0 (3.5% do dataset) contaminaria a normalização com zeros
# espúrios, distorcendo a escala de toda a feature.

def _eng_raw_masked(df, feature_cols, tip_values, tip_min=0.0, tip_max=50.0):
    """
    Filtra linhas com gorjeta válida e devolve (X_array, y_array).
    As features derivadas são calculadas sobre os valores brutos filtrados.
    Funciona quer o df tenha a coluna tip_amount quer não (usa tip_values).
    """
    tip = tip_values.astype(float)
    mask = (tip > tip_min) & (tip <= tip_max)

    df_m  = df[mask].reset_index(drop=True)
    y_m   = tip[mask]

    X_base       = df_m[feature_cols].values.astype(float)
    fare         = df_m["fare_amount"].values.astype(float)
    dist         = df_m["trip_distance"].values.astype(float)
    dur          = df_m["trip_duration_min"].values.astype(float)
    tip_m        = y_m  # usa y filtrado em vez de df["tip_amount"]

    # fare_per_minute: intensidade tarifária (proxy do tipo de viagem)
    fare_per_min = (fare / np.where(dur > 0, dur, np.nan))
    fare_per_min = np.nan_to_num(fare_per_min, nan=0.0).reshape(-1, 1)

    # log_distance: comprime a cauda longa das distâncias longas
    log_dist     = np.log1p(np.abs(dist)).reshape(-1, 1)

    # NOTA: tip_rate (tip_amount/fare_amount) foi removida — contém o target
    # directamente e causava data leakage (R²=0.97 artificialmente elevado).
    X_full = np.hstack([X_base, fare_per_min, log_dist])
    return X_full, y_m

# Verificar se os DataFrames brutos estão disponíveis (ramo splits CSV)
_has_raw_dfs = (
    "train_df" in dir() and train_df is not None and
    "val_df"   in dir() and val_df   is not None and
    "test_df"  in dir() and test_df  is not None and
    "tip_amount" in train_df.columns
)

if _has_raw_dfs:
    X_reg_tr_raw, y_reg_tr_ = _eng_raw_masked(train_df, FEATURE_COLS, y_reg_tr)
    X_reg_va_raw, y_reg_va_ = _eng_raw_masked(val_df,   FEATURE_COLS, y_reg_va)
    X_reg_te_raw, y_reg_te_ = _eng_raw_masked(test_df,  FEATURE_COLS, y_reg_te)
    # Normalizar: fit APENAS no treino (já filtrado), sem zeros de tip=0
    scaler_reg  = StandardScaler()
    X_reg_tr    = scaler_reg.fit_transform(X_reg_tr_raw)
    X_reg_va    = scaler_reg.transform(X_reg_va_raw)
    X_reg_te    = scaler_reg.transform(X_reg_te_raw)
    print("   ✓ Feature engineering sobre linhas com gorjeta (fare_per_min, log_dist) — tip_rate removida (leakage)")
else:
    # Fallback: usar os arrays normalizados base, apenas com máscara de tip > 0
    mask_tr = (y_reg_tr > 0) & (y_reg_tr <= 50)
    mask_va = (y_reg_va > 0) & (y_reg_va <= 50)
    mask_te = (y_reg_te > 0) & (y_reg_te <= 50)
    X_reg_tr    = X_train_sc[mask_tr]
    X_reg_va    = X_val_sc[mask_va]
    X_reg_te    = X_test_sc[mask_te]
    y_reg_tr_   = y_reg_tr[mask_tr]
    y_reg_va_   = y_reg_va[mask_va]
    y_reg_te_   = y_reg_te[mask_te]
    print("   ⚠ DataFrames brutos não disponíveis — a usar features base")

n_features_reg = X_reg_tr.shape[1]
print(f"   Features regressão (com eng.): {n_features_reg}")
print(f"   Amostras regressão — treino: {len(y_reg_tr_):,} | val: {len(y_reg_va_):,} | teste: {len(y_reg_te_):,}")

def build_deep_mlp_reg(n_feat, dropout=0.25, l2=5e-5):
    """
    Arquitetura mais profunda (256→128→64→32→16→1) com:
      - MSE como loss (conforme enunciado; RMSE e R² derivam diretamente)
      - BatchNorm + Dropout progressivo
      - ReduceLROnPlateau para ajuste dinâmico da LR
    """
    inputs = keras.Input(shape=(n_feat,))

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout / 2)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",   # MSE conforme enunciado (RMSE e R² derivam de MSE)
        metrics=["mae"]
    )
    return model

def get_reg_callbacks(patience=15):
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0),
    ]

deep_reg = build_deep_mlp_reg(n_features_reg)

history_reg = deep_reg.fit(
    X_reg_tr, y_reg_tr_,
    validation_data=(X_reg_va, y_reg_va_),
    epochs=EPOCHS_REG,
    batch_size=256,          # batch menor → gradientes mais informativos
    callbacks=get_reg_callbacks(patience=15),
    verbose=1
)

y_reg_pred = deep_reg.predict(X_reg_te, verbose=0).flatten()
# Garantir previsões não-negativas (gorjeta >= 0)
y_reg_pred = np.clip(y_reg_pred, 0, None)
rmse_dl = np.sqrt(mean_squared_error(y_reg_te_, y_reg_pred))
mae_dl  = mean_absolute_error(y_reg_te_, y_reg_pred)
r2_dl   = r2_score(y_reg_te_, y_reg_pred)

print(f"\n  Resultados Deep MLP Regressão:")
print(f"   RMSE : {rmse_dl:.4f}")
print(f"   MAE  : {mae_dl:.4f}")
print(f"   R²   : {r2_dl:.4f}")

salvar_regressao(y_reg_te_, y_reg_pred, SLUG_DEEP_MLP, "Deep MLP")

# SI: aprendizagem não supervisionada — Autoencoder para anomalias no espaço latente
# Spark: dados de entrada preparados na fase 1/2; esta fase corre em Python/TensorFlow
# ─────────────────────────────────────────
# MODELO 3 — AUTOENCODER (Não supervisionado)
# ─────────────────────────────────────────
print("\n── Autoencoder (Não Supervisionado) ──")
ENCODING_DIM = 4

def build_autoencoder(n_features, encoding_dim=4):
    inputs = keras.Input(shape=(n_features,), name="encoder_input")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    encoded = layers.Dense(encoding_dim, activation="relu", name="latent")(x)

    x = layers.Dense(16, activation="relu")(encoded)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    decoded = layers.Dense(n_features, activation="linear", name="output")(x)

    autoencoder = keras.Model(inputs, decoded, name="autoencoder")
    encoder = keras.Model(inputs, encoded, name="encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(n_features, ENCODING_DIM)
X_ae_train = X_train_sc[y_clf_tr == 1]
X_ae_val = X_val_sc[y_clf_va == 1]
X_ae_test = X_test_sc

history_ae = autoencoder.fit(
    X_ae_train, X_ae_train,
    validation_data=(X_ae_val, X_ae_val),
    epochs=20,
    batch_size=512,
    callbacks=get_callbacks(patience=8),
    verbose=1
)

X_recon = autoencoder.predict(X_ae_test, verbose=0)
recon_errors = np.mean(np.square(X_ae_test - X_recon), axis=1)
threshold = np.percentile(recon_errors, 95)
anomalies = recon_errors > threshold
latent = encoder.predict(X_test_sc, verbose=0)

print(f"   Limiar de anomalia (P95): {threshold:.4f}")
print(f"   Taxa de anomalias: {anomalies.mean()*100:.1f}%")

salvar_autoencoder_analise(history_ae, recon_errors, y_clf_te, threshold, latent)

# ─────────────────────────────────────────────────────────────────
# ABLATION STUDY — Deep MLP (Classificação tip_given)
# MELHORIA (feedback): treinar sistematicamente 3 variantes (A1, A2, A3)
# com crescente profundidade/complexidade e comparar numa tabela CSV.
# Demonstra que a escolha da arquitectura foi informada por evidência
# empírica e não arbitrária.
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ABLATION STUDY — Arquitecturas Deep MLP (tip_high)")
print("=" * 60)
# NOTA: O Ablation Study é feito sobre tip_high (25/75) em vez de tip_given (96.5% classe 1).
# tip_given colapsa para prever sempre 1 → todas as arquitecturas ficam iguais (inútil).
# tip_high tem desequilíbrio tratável e permite observar diferenças reais entre arquitecturas.
if not has_tip_high:
    print("   ⚠ tip_high não disponível — a saltar Ablation Study")
    ablation_results = []
else:
    # Recalcular class_weight para ablation (mesmo cálculo do treino principal)
    n_neg_ab = int(np.sum(y_hi_tr == 0))
    n_pos_ab = int(np.sum(y_hi_tr == 1))
    total_ab = n_neg_ab + n_pos_ab
    cw_ab = {0: total_ab / (2.0 * n_neg_ab), 1: total_ab / (2.0 * n_pos_ab)}

from sklearn.metrics import roc_auc_score as _roc_auc

def _build_ablation_model(arch_id, n_feat):
    inp = keras.Input(shape=(n_feat,))
    if arch_id == "A1":
        # A1: Simples, sem BatchNorm, Dropout=0.2
        x = layers.Dense(64, activation="relu")(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    elif arch_id == "A2":
        # A2: Media, BatchNorm em cada bloco, Dropout=0.3
        x = layers.Dense(128)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
    else:
        # A3: Profunda, 4 blocos, BatchNorm+Dropout progressivo
        x = layers.Dense(256)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

ablation_configs = [
    {"id": "A1", "desc": "Simples (64-32, sem BatchNorm, Dropout=0.2)"},
    {"id": "A2", "desc": "Media (128-64-32, BatchNorm, Dropout=0.3)"},
    {"id": "A3", "desc": "Profunda (256-128-64-32, BatchNorm, Dropout=0.3/0.4)"},
]

ablation_results = []
if has_tip_high:
    for cfg in ablation_configs:
        print(f"\n-- {cfg['id']}: {cfg['desc']} --")
        tf.random.set_seed(RANDOM_STATE)
        m = _build_ablation_model(cfg["id"], n_features)
        h = m.fit(
            X_train_sc, y_hi_tr,                    # ← tip_high
            validation_data=(X_val_sc, y_hi_va),
            epochs=30, batch_size=512,
            class_weight=cw_ab,                     # ← class_weight para evitar colapso
            callbacks=get_callbacks(patience=7),
            verbose=0,
        )
        n_epochs   = len(h.history["loss"])
        y_prob_ab  = m.predict(X_test_sc, verbose=0).flatten()
        y_pred_ab  = (y_prob_ab >= 0.5).astype(int)
        ab_acc     = float(accuracy_score(y_hi_te, y_pred_ab))
        ab_prec    = float(precision_score(y_hi_te, y_pred_ab, zero_division=0))
        ab_rec     = float(recall_score(y_hi_te, y_pred_ab, zero_division=0))
        ab_f1      = float(f1_score(y_hi_te, y_pred_ab, zero_division=0))
        ab_auc     = float(_roc_auc(y_hi_te, y_prob_ab))
        val_loss_f = float(h.history["val_loss"][-1])
        print(f"   Epocas: {n_epochs} | Acc={ab_acc:.4f} | F1={ab_f1:.4f} | AUC-ROC={ab_auc:.4f}")
        ablation_results.append({
            "Arquitectura":   cfg["id"],
            "Descricao":      cfg["desc"],
            "Epocas":         n_epochs,
            "Val_Loss_Final": round(val_loss_f, 4),
            "Accuracy":       round(ab_acc, 4),
            "Precision":      round(ab_prec, 4),
            "Recall":         round(ab_rec, 4),
            "F1-score":       round(ab_f1, 4),
            "AUC-ROC":        round(ab_auc, 4),
        })
else:
    print("   ⚠ tip_high não disponível — ablation_results vazio")

ablation_df = pd.DataFrame(ablation_results)
if not ablation_df.empty:
    ablation_csv_path = os.path.join(
        model_dir(SLUG_DEEP_MLP, TASK_TIP_HIGH), "ablation_estudo.csv",
    )
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"\n   ✓ {SLUG_DEEP_MLP}/{TASK_TIP_HIGH}/ablation_estudo.csv")

from spark_display import imprimir_pandas as _ip, secao as _sec
_sec("Ablation Study — Resultados Comparativos (Deep MLP Classificacao tip_high)")
_ip(ablation_df, dec=4)

salvar_ablation_estudo(ablation_results)

hp_results = ablation_results  # alias para compatibilidade com fase4

# ─────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────
dl_results = {
    "deep_mlp_clf": {
        "accuracy": deep_clf_metrics["accuracy"],
        "precision": deep_clf_metrics["precision"],
        "recall": deep_clf_metrics["recall"],
        "f1": deep_clf_metrics["f1"],
        "auc": deep_clf_metrics["auc"]
    },
    "deep_mlp_reg": {
        "rmse": rmse_dl, "mae": mae_dl, "r2": r2_dl
    },
    "autoencoder": {
        "reconstruction_threshold": float(threshold),
        "anomaly_rate": float(anomalies.mean())
    }
}

# ═════════════════════════════════════════════════════════════
# SI: KMeans e DBSCAN — comparação supervisionado vs não supervisionado (fase 4)
# Spark: clustering no espaço latente do autoencoder; métricas Silhueta/Davies-Bouldin
# MODELOS NÃO SUPERVISIONADOS CLÁSSICOS (para comparação)
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODELOS NÃO SUPERVISIONADOS CLÁSSICOS")
print("=" * 60)

print("\n── KMeans Clustering ──")
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_train_sc)
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X_train_sc, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_train_sc, kmeans_labels)
print(f"   Inércia: {kmeans_inertia:.4f}")
print(f"   Silhueta: {kmeans_silhouette:.4f}")
print(f"   Davies-Bouldin: {kmeans_davies_bouldin:.4f}")
kmeans_pred = kmeans.predict(X_test_sc)
kmeans_silhouette_test = silhouette_score(X_test_sc, kmeans_pred)
print(f"   Silhueta (teste): {kmeans_silhouette_test:.4f}")
salvar_clusters_pca(X_test_sc, kmeans_pred, SLUG_KMEANS, "KMeans", color_by=y_clf_te)

print("\n── DBSCAN Clustering ──")
# DBSCAN requer eps ajustado para dados normalizados
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_train_sc)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"   Clusters encontrados: {n_clusters}")
print(f"   Pontos de ruído: {n_noise} ({100*n_noise/len(dbscan_labels):.1f}%)")
if n_clusters > 1 and n_noise < len(dbscan_labels):
    dbscan_silhouette = silhouette_score(X_train_sc[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
    print(f"   Silhueta (sem ruído): {dbscan_silhouette:.4f}")
dbscan_pred = dbscan.fit_predict(X_test_sc)
n_clusters_test = len(set(dbscan_pred)) - (1 if -1 in dbscan_pred else 0)
print(f"   Clusters no teste: {n_clusters_test}")
salvar_clusters_pca(X_test_sc, dbscan_pred, SLUG_DBSCAN, "DBSCAN", color_by=y_clf_te)

# Consolidar resultados não supervisionados
unsupervised_results = pd.DataFrame([
    {"Modelo": "KMeans", "Silhueta_Treino": f"{kmeans_silhouette:.4f}", "Silhueta_Teste": f"{kmeans_silhouette_test:.4f}", "Davies_Bouldin": f"{kmeans_davies_bouldin:.4f}"},
    {"Modelo": "DBSCAN", "Clusters": n_clusters, "Ruído_%": f"{100*n_noise/len(dbscan_labels):.1f}%", "Clusters_Teste": n_clusters_test},
    {"Modelo": "Autoencoder", "Limiar_P95": f"{threshold:.4f}", "Taxa_Anomalia_%": f"{anomalies.mean()*100:.1f}%", "n_anomalias": int(anomalies.sum())},
])
imprimir_pandas(unsupervised_results, dec=4, titulo="Resumo Modelos Não Supervisionados")
unsupervised_results.to_csv(os.path.join(OUTPUT_DIR, "unsupervised_models_results.csv"), index=False)
print("   ✓ unsupervised_models_results.csv")

dl_results = {
    "deep_mlp_clf": {
        "accuracy": float(deep_clf_metrics['accuracy']),
        "precision": float(deep_clf_metrics['precision']),
        "recall": float(deep_clf_metrics['recall']),
        "f1": float(deep_clf_metrics['f1']),
        "auc": float(deep_clf_metrics['auc'])
    },
    "deep_mlp_clf_tip_high": deep_clf_high_metrics,
    "deep_mlp_reg": {
        "rmse": rmse_dl, "mae": mae_dl, "r2": r2_dl
    },
    "autoencoder": {
        "reconstruction_threshold": float(threshold),
        "anomaly_rate": float(anomalies.mean())
    },
    "unsupervised_classifiers": {
        "kmeans": {"silhouette_train": float(kmeans_silhouette), "silhouette_test": float(kmeans_silhouette_test)},
        "dbscan": {"clusters_train": int(n_clusters), "noise_pct": float(100*n_noise/len(dbscan_labels))}
    }
}

import json
with open(f"{OUTPUT_DIR}/resultados_deep_learning.json", "w") as f:
    json.dump(dl_results, f, indent=2)

print("\n" + "=" * 60)
print("RESUMO DEEP LEARNING")
print("=" * 60)
print(f"\n  Deep MLP Classificação:")
print(
    f"   Accuracy={deep_clf_metrics['accuracy']:.4f} | "
    f"F1={deep_clf_metrics['f1']:.4f} | "
    f"AUC={deep_clf_metrics['auc']:.4f}"
)
print(f"\n  Deep MLP Regressão:")
print(f"   RMSE={rmse_dl:.4f} | MAE={mae_dl:.4f} | R²={r2_dl:.4f}")
print(f"\n  Autoencoder:")
print(f"   Limiar anomalia={threshold:.4f} | Taxa={anomalies.mean()*100:.1f}%")

print("\n✅ Fase 3 concluída! Próximo passo: fase4_comparacao.py")