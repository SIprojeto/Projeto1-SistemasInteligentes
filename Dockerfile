# ─────────────────────────────────────────────────────────────
#  Imagem base: Python 3.11 slim (Debian Bookworm)
#  Inclui Java 17 (obrigatório para PySpark) + todas as deps
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# ── Metadata ──────────────────────────────────────────────────
LABEL maintainer="projeto-si-bigdata"
LABEL description="NYC Taxi Tip Prediction — SI + Big Data pipeline"

# ── Variáveis de ambiente ─────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3 \
    # Evita que o Spark tente abrir UI na porta 4040 (opcional)
    SPARK_NO_DAEMONIZE=true

# ── Sistema: Java 17 + utilitários ───────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Directório de trabalho ────────────────────────────────────
WORKDIR /app

# ── Dependências Python ───────────────────────────────────────
# Copia só o requirements primeiro para aproveitar cache do Docker
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt

# ── Código do projecto ────────────────────────────────────────
COPY . .

# ── Directórios de output (criados em runtime) ───────────────
RUN mkdir -p outputs/figures outputs/reports outputs/models outputs/splits data

# ── Porta do Spark UI (opcional, só para debug) ───────────────
EXPOSE 4040

# ── Entrypoint ───────────────────────────────────────────────
# Por omissão corre todas as fases; passa --fase 1 2 etc. para seleccionar
CMD ["python", "main.py"]
