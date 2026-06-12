# SI: orquestrador end-to-end das 4 fases (ML clássico, DL, comparação) com seeds fixos
# Spark: reprodutibilidade do pipeline BD Stages 1–8 via fase 1 + fase 2 Spark ML
"""
Orquestrador do Projeto — Sistemas Inteligentes + Big Data & Data Mining
NYC Yellow Taxi Trip Data — Previsão de Gorjetas

Uso:
    python main.py              # Executa todas as fases
    python main.py --fase 2     # Executa apenas a fase 2
    python main.py --fase 1 2   # Executa fases 1 e 2
    python main.py --check      # Verifica dependências sem executar
    python main.py --demo       # Apresentação: salta fases prontas + tuning rápido
"""

import os
import sys
import subprocess
import time
import argparse
import platform
import importlib.util
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FASES = [
    {
        "id": 1,
        "script": os.path.join("src", "fase1_eda.py"),
        "descricao": "Análise Exploratória de Dados (EDA) com Spark",
        "outputs_esperados": [
            "outputs/reports/dataset_processado.csv",
            "outputs/reports/tabela_summary.csv",
            "outputs/fase1_eda/fig1_distribuicoes.png",
            "outputs/fase1_eda/fig3_correlacao.png",
        ],
    },
    {
        "id": 2,
        "script": os.path.join("src", "fase2", "fase2_runner.py"),
        "descricao": "Modelos ML clássicos (Spark ML + Sklearn)",
        "outputs_esperados": [
            "outputs/reports/resultados_classificacao.csv",
            "outputs/reports/sklearn_classification_results.csv",
            "outputs/models/spark_rf_classifier",
            "outputs/models/spark_lr_classifier",
            "outputs/splits/train.parquet",
            "outputs/splits/validation.parquet",
            "outputs/splits/test.parquet",
            "outputs/fase2_ml/random_forest/classificacao_tip_given/matriz_confusao.png",
            "outputs/fase2_ml/svm/classificacao_tip_given/curvas_roc_pr.png",
            "outputs/fase2_ml/j48/classificacao_tip_given/arvore_decisao.png",
            "outputs/fase2_ml/mlp_sklearn/regressao_tip_amount/previsto_vs_real.png",
        ],
    },
    {
        "id": 3,
        "script": os.path.join("src", "fase3_deep_learning.py"),
        "descricao": "Deep Learning (TensorFlow/Keras)",
        "outputs_esperados": [
            "outputs/reports/resultados_deep_learning.json",
            "outputs/fase3_dl_anomalias/deep_mlp/classificacao_tip_given/matriz_confusao.png",
            "outputs/fase3_dl_anomalias/deep_mlp/regressao_tip_amount/previsto_vs_real.png",
            "outputs/fase3_dl_anomalias/autoencoder/distribuicao_erro.png",
            "outputs/fase3_dl_anomalias/deep_mlp/classificacao_tip_high/ablation_estudo.png",
        ],
    },
    {
        "id": 4,
        "script": os.path.join("src", "fase4_comparacao.py"),
        "descricao": "Comparação Final ML vs Deep Learning",
        "outputs_esperados": [
            "outputs/reports/comparacao_classificacao.csv",
            "outputs/reports/comparacao_regressao.csv",
            "outputs/fase4_comparacao/fig_comparacao_redes_neuronais.png",
            "outputs/fase4_comparacao/fig_comparacao_rn_tip_high.png",
            "outputs/fase4_comparacao/fig_comparacao_rn_tip_amount.png",
        ],
    },
]

DEPENDENCIAS = {
    "numpy":      "numpy",
    "pandas":     "pandas",
    "matplotlib": "matplotlib",
    "seaborn":    "seaborn",
    "sklearn":    "scikit-learn",
    "tensorflow": "tensorflow",
    "pyspark":    "pyspark",
}

DATASET_FILENAME = "yellow_tripdata_2016-03.csv"
SEP = "=" * 62


def _cor(texto, codigo):
    if platform.system() == "Windows" and "ANSICON" not in os.environ:
        return texto
    return f"\033[{codigo}m{texto}\033[0m"


def ok(msg):   print(_cor(f"   ✓ {msg}", "32"))
def warn(msg): print(_cor(f"   ⚠ {msg}", "33"))
def erro(msg): print(_cor(f"   ✗ {msg}", "31"))
def info(msg): print(f"   · {msg}")


def formatar_tempo(segundos):
    if segundos < 60:
        return f"{segundos:.1f}s"
    m, s = divmod(int(segundos), 60)
    return f"{m}m {s}s"


def verificar_dependencias():
    print(f"\n{SEP}")
    print("  VERIFICAÇÃO DE DEPENDÊNCIAS")
    print(SEP)
    tudo_ok = True
    for modulo, pacote in DEPENDENCIAS.items():
        spec = importlib.util.find_spec(modulo)
        if spec is None:
            erro(f"{pacote} NÃO instalado  →  pip install {pacote}")
            tudo_ok = False
        else:
            try:
                mod = importlib.import_module(modulo)
                versao = getattr(mod, "__version__", "?")
                ok(f"{pacote} {versao}")
            except Exception:
                ok(f"{pacote} (versão indisponível)")

    java_ok = os.system("java -version >nul 2>&1" if platform.system() == "Windows"
                        else "java -version >/dev/null 2>&1") == 0
    if java_ok:
        ok("Java  (obrigatório para PySpark)")
    else:
        erro("Java NÃO encontrado — instale Java 11+")
        tudo_ok = False
    return tudo_ok


def verificar_dataset():
    dataset_path = os.path.join(BASE_DIR, "data", DATASET_FILENAME)
    if os.path.exists(dataset_path):
        tamanho_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        ok(f"Dataset encontrado: {DATASET_FILENAME}  ({tamanho_mb:.0f} MB)")
        return True
    else:
        erro(f"Dataset NÃO encontrado em: data/{DATASET_FILENAME}")
        info("Descarregue em: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data")
        info(f"Coloque o ficheiro em: {os.path.join(BASE_DIR, 'data')}/")
        return False


def verificar_outputs_fase(fase):
    existentes, em_falta = [], []
    for output in fase["outputs_esperados"]:
        path = os.path.join(BASE_DIR, output)
        (existentes if os.path.exists(path) else em_falta).append(output)
    return existentes, em_falta


def limpar_nao_figuras():
    """
    Remove artefactos tabulares/textuais para entrega focada em figuras.
    Mantém apenas conteúdos de outputs/figures e modelos guardados.
    """
    alvos = [
        os.path.join(BASE_DIR, "outputs", "reports"),
        os.path.join(BASE_DIR, "outputs", "splits"),
        os.path.join(BASE_DIR, "outputs", "models"),
    ]
    exts_remover = {".csv", ".txt", ".json", ".parquet"}
    # Manter só o dataset para Weka/reprodutibilidade; resto de reports é substituído por figuras
    manter_em_reports = {"dataset_processado.csv"}
    removidos = 0
    for alvo in alvos:
        p = Path(alvo)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts_remover:
                if "reports" in str(f).replace("\\", "/") and f.name in manter_em_reports:
                    continue
                try:
                    f.unlink()
                    removidos += 1
                except Exception:
                    pass
    if removidos > 0:
        ok(f"Limpeza figures-only: {removidos} ficheiros removidos em outputs/reports|splits")
    else:
        info("Limpeza figures-only: sem ficheiros para remover")


# SI: execução modular por fase e validação dos outputs esperados no relatório
# Spark: fases 1–2 cumprem ingestão RDD, SQL, pipeline e tuning exigidos em bigDataSpark.txt
def executar_fase(fase):
    fid = fase["id"]
    script_path = os.path.join(BASE_DIR, fase["script"])

    print(f"\n{SEP}")
    print(f"  FASE {fid} — {fase['descricao']}")
    print(SEP)

    if not os.path.exists(script_path):
        erro(f"Script não encontrado: {fase['script']}")
        raise FileNotFoundError(script_path)

    inicio = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    duracao = time.time() - inicio

    print()
    if result.returncode == 0:
        ok(f"Fase {fid} concluída em {formatar_tempo(duracao)}")
        _, ainda_em_falta = verificar_outputs_fase(fase)
        if ainda_em_falta:
            warn(f"Outputs em falta ({len(ainda_em_falta)}): {ainda_em_falta[:3]}")
        else:
            ok(f"Todos os outputs verificados ({len(fase['outputs_esperados'])})")
    else:
        erro(f"Fase {fid} falhou (código {result.returncode}) ao fim de {formatar_tempo(duracao)}")
        raise SystemExit(result.returncode)

    return duracao


def main():
    parser = argparse.ArgumentParser(description="Orquestrador — Projeto SI + Big Data")
    parser.add_argument("--fase", nargs="+", type=int, choices=[1, 2, 3, 4], metavar="N")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Saltar fases cujos outputs esperados já existem (útil na apresentação)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Modo apresentação: --skip-complete + SKIP_IF_READY + FAST_TUNING",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="No fim, remove CSV/TXT/JSON/PARQUET em outputs/reports e outputs/splits",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Não escrever CSV/TXT/JSON em outputs/reports (figuras continuam).",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Não guardar modelos/pipeline em outputs/models.",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Não persistir splits em outputs/splits.",
    )
    args = parser.parse_args()

    if args.no_reports:
        os.environ["WRITE_REPORTS"] = "0"
    if args.no_models:
        os.environ["WRITE_MODELS"] = "0"
    if args.no_splits:
        os.environ["WRITE_SPLITS"] = "0"

    if args.demo:
        os.environ.setdefault("SKIP_IF_READY", "1")
        os.environ.setdefault("FAST_TUNING", "1")
        args.skip_complete = True

    print(f"\n{SEP}")
    print("  ORQUESTRADOR — Projeto SI + Big Data")
    print(f"  Python {sys.version.split()[0]}  |  {platform.system()} {platform.machine()}")
    print(SEP)

    if not args.skip_check:
        deps_ok = verificar_dependencias()
        dataset_ok = verificar_dataset()
        if not deps_ok or not dataset_ok:
            erro("Resolva os problemas acima antes de executar.")
            sys.exit(1)

    if args.check:
        ok("Verificação concluída.")
        sys.exit(0)

    ids_a_executar = args.fase if args.fase else [f["id"] for f in FASES]
    fases_a_executar = [f for f in FASES if f["id"] in ids_a_executar]

    print(f"\n   Fases a executar: {[f['id'] for f in fases_a_executar]}")

    tempos = {}
    inicio_total = time.time()
    for fase in fases_a_executar:
        if args.skip_complete:
            _, em_falta = verificar_outputs_fase(fase)
            if not em_falta:
                ok(f"Fase {fase['id']} já completa — a saltar ({len(fase['outputs_esperados'])} outputs)")
                tempos[fase["id"]] = 0.0
                continue
        tempos[fase["id"]] = executar_fase(fase)
    total = time.time() - inicio_total

    print(f"\n{SEP}")
    print("  RESUMO DE TEMPOS")
    print(SEP)
    for fid, dur in tempos.items():
        desc = next(f["descricao"] for f in FASES if f["id"] == fid)
        print(f"   Fase {fid}: {formatar_tempo(dur):>8}  — {desc}")
    print(f"   {'TOTAL':6}: {formatar_tempo(total):>8}")
    print(SEP)
    if args.figures_only:
        limpar_nao_figuras()
    ok("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
