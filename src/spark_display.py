"""
Helpers para imprimir DataFrames Spark/pandas no terminal sem partir colunas largas.
Todas as tabelas: moldura com bordas horizontais e verticais (box-drawing Unicode).
"""

import sys

import pandas as pd

LARGURA = 72
LARGURA_CELULA = 40

_BOX_UNICODE = {
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "h": "─",
    "v": "│",
    "tm": "┬",
    "bm": "┴",
    "lm": "├",
    "rm": "┤",
    "cross": "┼",
}

_BOX_ASCII = {
    "tl": "+",
    "tr": "+",
    "bl": "+",
    "br": "+",
    "h": "-",
    "v": "|",
    "tm": "+",
    "bm": "+",
    "lm": "+",
    "rm": "+",
    "cross": "+",
}


def _stdout_suporta_unicode():
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if enc in ("utf-8", "utf8", "cp65001"):
        return True
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            return True
        except Exception:
            pass
    return False


_BOX = _BOX_UNICODE if _stdout_suporta_unicode() else _BOX_ASCII


def linha_tracejada(largura=None, char=None):
    """Separador tracejado (secções); usa o mesmo traço das tabelas."""
    n = min(largura or LARGURA, LARGURA)
    print((char or _BOX["h"]) * max(n, 20))


def secao(titulo=None, sublinhado=False):
    """Bloco visual: linha em branco + título opcional."""
    print()
    if titulo:
        print(titulo)
        if sublinhado:
            linha_tracejada(min(len(titulo) + 4, LARGURA))
        print()


def formatar_vetor(v, max_preview=3, dec=3):
    if v is None:
        return "null"
    try:
        arr = v.toArray() if hasattr(v, "toArray") else list(v)
    except Exception:
        return str(v)[:48]
    n = len(arr)
    if n == 0:
        return "[]"
    if n <= max_preview:
        inner = ", ".join(f"{x:.{dec}f}" for x in arr)
        return f"[{inner}] ({n}d)"
    inner = ", ".join(f"{x:.{dec}f}" for x in arr[:max_preview])
    return f"[{inner}, ...] ({n} dims)"


def _format_stats_table(pdf, dec=3):
    """Transpõe tabelas summary/describe (linhas=estatísticas) para variável × métrica."""
    if "summary" not in pdf.columns:
        return pdf
    out = pdf.set_index("summary").T.reset_index(names="variavel")
    for col in out.columns:
        if col == "variavel":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").round(dec)
    return out


def _truncar(texto, largura):
    s = str(texto)
    if len(s) <= largura:
        return s
    if largura <= 1:
        return s[:largura]
    return s[: largura - 1] + "…"


def _larguras_colunas(headers, rows, col_space=10, max_w=LARGURA_CELULA):
    n = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(n):
            widths[i] = max(widths[i], len(str(row[i])))
    min_w = max(1, col_space - 2)
    return [min(max(w, min_w), max_w) for w in widths]


def _linha_horizontal(widths, left, mid, right):
    partes = [_BOX["h"] * (w + 2) for w in widths]
    return left + mid.join(partes) + right


def _linha_dados(cells, widths, aligns):
    partes = []
    for cell, w, al in zip(cells, widths, aligns):
        txt = _truncar(cell, w)
        partes.append(f" {txt:{al}{w}} ")
    return _BOX["v"] + _BOX["v"].join(partes) + _BOX["v"]


def _alinhamentos(pdf):
    return [
        ">" if pd.api.types.is_numeric_dtype(pdf[c]) else "<"
        for c in pdf.columns
    ]


def _imprimir_caixa_vazia(mensagem="(sem resultados)"):
    msg = f" {mensagem} "
    w = len(msg)
    print(_BOX["tl"] + _BOX["h"] * w + _BOX["tr"])
    print(_BOX["v"] + msg + _BOX["v"])
    print(_BOX["bl"] + _BOX["h"] * w + _BOX["br"])


def _print_pandas(pdf, col_space=10, titulo=None):
    """
    Imprime tabela pandas com moldura completa:
      - bordas superior/inferior e laterais
      - separador entre cabeçalho e dados
      - divisórias verticais entre colunas
    """
    if titulo:
        secao(titulo)
    else:
        print()

    if pdf.empty:
        _imprimir_caixa_vazia()
        print()
        return

    headers = [str(c) for c in pdf.columns]
    rows = [[str(x) for x in r] for r in pdf.values.tolist()]
    if not rows:
        _imprimir_caixa_vazia()
        print()
        return

    widths = _larguras_colunas(headers, rows, col_space=col_space)
    aligns = _alinhamentos(pdf)
    header_aligns = ["<"] * len(headers)

    print(_linha_horizontal(widths, _BOX["tl"], _BOX["tm"], _BOX["tr"]))
    print(_linha_dados(headers, widths, header_aligns))
    print(_linha_horizontal(widths, _BOX["lm"], _BOX["cross"], _BOX["rm"]))
    for row in rows:
        print(_linha_dados(row, widths, aligns))
    print(_linha_horizontal(widths, _BOX["bl"], _BOX["bm"], _BOX["br"]))
    print()


def imprimir_amostra_spark(df, n=5, cols=None, cols_vetor=("features",), titulo=None):
    """Amostra de DataFrame Spark com vetores compactos."""
    from pyspark.sql import functions as _F
    sel = df.select(*cols) if cols else df
    pdf = sel.orderBy(_F.rand(seed=42)).limit(n).toPandas()
    for col in cols_vetor:
        if col in pdf.columns:
            pdf[col] = pdf[col].apply(formatar_vetor)
    _print_pandas(pdf, titulo=titulo)


def imprimir_spark_df(df, n=20, titulo=None):
    """Substitui .show() — tabela alinhada via pandas."""
    pdf = df.limit(n).toPandas()
    if pdf.empty:
        secao(titulo) if titulo else print()
        _imprimir_caixa_vazia()
        print()
        return
    if "summary" in pdf.columns:
        pdf = _format_stats_table(pdf)
    _print_pandas(pdf, titulo=titulo)


def imprimir_describe_spark(df, dec=3, titulo=None):
    """Substitui describe().show() — transposta (variável × estatística)."""
    _print_pandas(_format_stats_table(df.describe().toPandas(), dec), col_space=10, titulo=titulo)


def imprimir_describe_pandas(df, dec=3, titulo=None):
    """describe() de DataFrame pandas — transposta e arredondada."""
    stats = df.describe().round(dec).T.reset_index(names="variavel")
    _print_pandas(stats, col_space=10, titulo=titulo)


def imprimir_pandas(df, dec=4, na="—", titulo=None):
    """Tabela pandas legível (métricas, matrizes, resumos)."""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].apply(lambda x: f"{x:.{dec}f}" if pd.notna(x) else na)
        else:
            out[col] = out[col].fillna(na).astype(str)
    _print_pandas(out, col_space=11, titulo=titulo)
