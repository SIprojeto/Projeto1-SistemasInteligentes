"""
Compatibilidade — delega para model_plots (estrutura padronizada em outputs/modelos/).
"""

from model_plots import salvar_classificacao


def gerar_figuras_sklearn(y_true, y_pred, y_proba, model_slug, figures_dir=None, label_col="tip_given"):
    """
    Gera matriz de confusão + ROC/PR para modelos sklearn.

    ``model_slug`` deve ser um identificador estável (ex: svm, j48, mlp_sklearn).
    O parâmetro ``figures_dir`` é ignorado (mantido por compatibilidade).
    """
    titulo = model_slug.replace("_", " ").title()
    salvar_classificacao(y_true, y_pred, y_proba, model_slug, titulo, label_col)
