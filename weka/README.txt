================================================================================
  NYC Taxi Pipeline — Docker (docker-compose.yml)
  Pasta: projeto_final
================================================================================

NÃO uses no Windows (fora do Docker):
  python main.py
  pip install ...

Com Docker instalas só: Docker Desktop + dataset em data\

================================================================================
  O QUE É O DOCKER AQUI
================================================================================

  Dockerfile     → “máquina” com Python 3.11, Java 17, Spark, TensorFlow, etc.
  docker-compose → como correr essa máquina ligada à tua pasta do projeto

  Volumes (pastas partilhadas PC ↔ container):
    ./data      → CSV do Kaggle (tu pões o ficheiro)
    ./outputs   → gráficos, CSVs, modelos (ficam no teu disco)
    ./src       → código (editas no PC, corre no container)
    ./main.py   → orquestrador

  Ou seja: o container corre o pipeline; os resultados aparecem em outputs\

================================================================================
  ANTES DE COMEÇAR
================================================================================

  1. Docker Desktop aberto
  2. Dataset em:
       data\yellow_tripdata_2016-03.csv
     (~1.8 GB — Kaggle NYC Yellow Taxi, março 2016)
  3. Na pasta do projeto (onde está docker-compose.yml):

       cd c:\Users\marta\Desktop\projeto_final\magicproject\projeto_final

================================================================================
  PRIMEIRA VEZ — construir imagem + correr tudo
================================================================================

  docker compose up --build

  O que acontece:
  - build: lê Dockerfile + requirements.txt (demora na 1ª vez)
  - up: corre python main.py --skip-check (fases 1 → 2 → 3 → 4)
  - Tempo: ~15–30 min (Fase 1 lê o CSV grande; depois ML + DL)

  Quando acabar, vê:
    outputs\figures\
    outputs\reports\
    outputs\models\
    outputs\splits\

================================================================================
  DEPOIS DA PRIMEIRA VEZ — sem rebuild
================================================================================

  Só mudaste código (.py):

    docker compose run --rm pipeline python main.py --skip-check

  Mudaste Dockerfile ou requirements.txt:

    docker compose up --build

================================================================================
  CORRER UMA FASE SÓ
================================================================================

  docker compose run --rm pipeline python main.py --fase 1 --skip-check
  docker compose run --rm pipeline python main.py --fase 2 --skip-check
  docker compose run --rm pipeline python main.py --fase 3 --skip-check
  docker compose run --rm pipeline python main.py --fase 4 --skip-check

  Várias fases:

    docker compose run --rm pipeline python main.py --fase 1 2 --skip-check

================================================================================
  ATALHOS ÚTEIS
================================================================================

  Verificar dependências + dataset (não corre o pipeline):

    docker compose run --rm pipeline python main.py --check

  Saltar fases que já têm outputs (--demo = skip + tuning rápido):

    docker compose run --rm pipeline python main.py --demo --skip-check

  Teste rápido (grelhas Spark menores):

    set FAST_TUNING=1
    docker compose run --rm pipeline python main.py --fase 2 --skip-check

  Spark UI (enquanto corre fase 1 ou 2):
    http://localhost:4040

================================================================================
  APRESENTAÇÃO / DEFESA
================================================================================

  Se outputs\ já está completo (gerado em casa):
    → NÃO precisas de correr Docker na sala
    → Mostra outputs\figures e outputs\reports + relatório

  Se quiseres mostrar Docker na mesma:
    docker compose run --rm pipeline python main.py --demo --skip-check
    (segundos se outputs já existem)

================================================================================
  PROBLEMAS COMUNS
================================================================================

  “Dataset não encontrado”
    → Falta data\yellow_tripdata_2016-03.csv

  Container morre por memória
    → Corre fase a fase (--fase 1, depois 2, etc.)
    → Fecha outros programas pesados

  Parar containers:

    docker compose down

================================================================================
  RESUMO
================================================================================

  1ª vez     →  docker compose up --build
  Repetir    →  docker compose run --rm pipeline python main.py --skip-check
  Uma fase   →  ... python main.py --fase N --skip-check
  Defesa     →  pasta outputs\ (Docker opcional)
