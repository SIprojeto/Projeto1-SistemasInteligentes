# Projeto SI + Big Data — NYC Yellow Taxi Tip Prediction

## Estrutura do Projeto

```
projeto/
├── main.py                        ← Orquestrador principal
├── requirements.txt               ← Dependências Python
├── README.md                      ← Este ficheiro
│
├── data/
│   └── yellow_tripdata_2016-03.csv  ← Dataset (descarregar do Kaggle)
│
├── src/
│   ├── fase1_eda.py               ← EDA e limpeza com Spark
│   ├── fase2/
│   │   ├── spark_setup.py         ← Configurações partilhadas (Spark, paths, schema)
│   │   ├── spark_pipeline.py      ← Splits + Spark ML Pipeline
│   │   ├── random_forest.py       ← Random Forest (Spark ML)
│   │   ├── logistic_regression.py ← Logistic Regression (Spark ML)
│   │   ├── svm.py                 ← SVM classifier + regressor (sklearn)
│   │   ├── decision_tree.py       ← Decision Tree J48 (sklearn)
│   │   ├── mlp.py                 ← MLP classifier + regressor (sklearn)
│   │   ├── naive_bayes.py         ← Naive Bayes (sklearn)
│   │   ├── error_analysis.py      ← Threshold analysis + SQL error queries
│   │   └── fase2_runner.py        ← Runner da fase 2 (chamado pelo main.py)
│   ├── fase3_deep_learning.py     ← Deep MLP + Autoencoder (TensorFlow)
│   └── fase4_comparacao.py        ← Comparação ML vs Deep Learning
│
├── outputs/
│   ├── modelos/                   ← Figuras padronizadas por modelo (ver abaixo)
│   ├── fase1_eda/                 ← Gráficos da EDA
│   ├── fase4_comparacao/          ← Gráficos comparativos finais
│   ├── models/                    ← Modelos guardados (Spark ML, Keras)
│   ├── splits/                    ← train/val/test em Parquet
│   └── reports/                   ← CSVs, JSONs e TXTs com resultados
│
└── relatorio/
    └── relatorio_final.pdf        ← Relatório final (15-20 páginas)
```

## Como Executar

### Docker (recomendado no Windows)

Requisitos: Docker Desktop + dataset em `data/`. Guia completo: `projeto/para_correr_proj.txt`.

```bash
docker compose up --build          # primeira vez: imagem + pipeline completo
docker compose run --rm pipeline python main.py --skip-check   # repetir sem rebuild
docker compose run --rm pipeline python main.py --fase 2 --skip-check
```

Resultados em `outputs/`. Spark UI: http://localhost:4040

### Local (Python instalado no PC)

### 1. Pré-requisitos
- Python 3.9+
- Java 11+ (obrigatório para PySpark)
- Dataset do Kaggle: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Colocar o dataset
```
data/yellow_tripdata_2016-03.csv
```

### 4. Verificar instalação
```bash
python main.py --check
```

### 5. Executar tudo
```bash
python main.py
```

### 6. Executar apenas uma fase
```bash
python main.py --fase 1        # Só EDA
python main.py --fase 2        # Só modelos ML
python main.py --fase 1 2      # EDA + ML
python main.py --fase 3        # Só Deep Learning
python main.py --fase 4        # Só comparação final
```

## Fases do Projeto

| Fase | Ficheiro | O que faz |
|------|----------|-----------|
| 1 | `src/fase1_eda.py` | Carrega dados com RDD, limpa, gera EDA com gráficos |
| 2 | `src/fase2/fase2_runner.py` | SVM, Decision Tree J48, MLP, Naive Bayes, RF, LR |
| 3 | `src/fase3_deep_learning.py` | Deep MLP + Autoencoder com ajuste de hiperparâmetros |
| 4 | `src/fase4_comparacao.py` | Comparação final ML vs Deep Learning |

## Modelos Implementados

### Aprendizagem Supervisionada (Machine Learning)
- **SVM** — Support Vector Machine (kernel RBF, sklearn)
- **Decision Tree J48** — critério entropia, equivalente ao Weka J48 (sklearn)
- **Random Forest** — 50-150 árvores, tuning com TrainValidationSplit (Spark ML)
- **Logistic Regression** — regularização ElasticNet (Spark ML)
- **MLP** — 128-64-32, ReLU, Adam, early stopping (sklearn)
- **Naive Bayes** — GaussianNB (sklearn)

### Aprendizagem Profunda (Deep Learning)
- **Deep MLP** — classificação e regressão, BatchNorm, Dropout, L2
- **Autoencoder** — encoding_dim=4, análise do espaço latente

### Aprendizagem Não Supervisionada
- **KMeans** — clustering no espaço latente do autoencoder
- **DBSCAN** — clustering baseado em densidade

## Outputs Gerados

### Figuras padronizadas por modelo (`outputs/modelos/`)

Dois paradigmas no topo — **supervisionado** vs **não supervisionado** — e dentro de cada um os modelos com os mesmos ficheiros:

```
outputs/modelos/
  supervisionado/
    random_forest/
      classificacao_tip_given/
        matriz_confusao.png
        curvas_roc_pr.png
      classificacao_tip_high/   …
      regressao_tip_amount/
        previsto_vs_real.png
        residuos.png
    logistic_regression/  svm/  j48/  mlp_sklearn/  naive_bayes/  deep_mlp/
  nao_supervisionado/
    autoencoder/
      distribuicao_erro.png
      espaco_latente.png
    kmeans/
      clusters_pca.png
    dbscan/
      clusters_pca.png
```

**Classificação (supervisionado):** `matriz_confusao.png` + `curvas_roc_pr.png`  
**Regressão (supervisionado):** `previsto_vs_real.png` + `residuos.png`  
**J48 extra:** `arvore_decisao.png` na pasta de classificação  
**Deep MLP extra:** `ablation_estudo.png` em `classificacao_tip_high/`

### Tamanho do dataset

Por defeito usa 2M linhas. Para correr com **300k**:

```bash
# Windows PowerShell
$env:DATASET_N_ROWS="300000"; python main.py

# Linux / macOS / Git Bash
DATASET_N_ROWS=300000 python main.py
```

Após execução completa, também em `outputs/`:

```
fase1_eda/
  fig1_distribuicoes.png         ← Histogramas das variáveis
  fig3_correlacao.png            ← Matriz de correlação

fase4_comparacao/
  fig15_radar_classificacao.png  ← Radar chart comparativo
  fig16_comparacao_completa.png  ← ML vs DL (resumo)
  ...

reports/
  resultados_classificacao.csv   ← Métricas Spark ML (RF, LR)
  sklearn_classification_results.csv  ← Métricas sklearn
  resultados_deep_learning.json  ← Resultados Deep Learning
  comparacao_classificacao.csv   ← Tabela final comparativa
  domain_analysis_stage8.txt     ← Análise FP vs FN (Stage 8 BD)
  threshold_recommendation.txt   ← Threshold selecionado

splits/
  train.parquet                  ← 60% treino (Parquet)
  validation.parquet             ← 20% validação (Parquet)
  test.parquet                   ← 20% teste (Parquet)

models/
  spark_pipeline/                ← Pipeline Spark ML
  spark_rf_classifier/           ← Melhor RF classifier
  spark_lr_classifier/           ← Melhor LR classifier
  deep_mlp_clf/                  ← Deep MLP para classificação
  autoencoder/                   ← Autoencoder treinado
```

## Notas Técnicas

- **Seed fixo**: 42 em todos os modelos (reprodutibilidade total)
- **Data leakage**: pipeline fit apenas em treino, transform em val/test
- **Schema explícito**: StructType em todos os reads (sem inferSchema)
- **SVM subsample**: 3000 amostras (O(n²) memory constraint — justificado)
- **Métrica de tuning**: PR-AUC (adequada para dados desequilibrados)
- **Constraint de domínio**: recall ≥ 0.70 (FN mais custoso que FP)
