import pandas as pd

from sklearn.model_selection import cross_validate, GridSearchCV

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


RANDOM_STATE = 42


def construir_pipeline_modelo_classificacao(classificador, preprocessor=None):
    if preprocessor is not None:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("clf", classificador)
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("clf", classificador)
            ]
        )

    model = pipeline

    return model


def treinar_e_validar_modelo_classificacao(
    X,
    y,
    cv,
    classificador,
    preprocessor=None,
):

    model = construir_pipeline_modelo_classificacao(
        classificador,
        preprocessor,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
    )

    return scores


def grid_search_cv_classificador(
    classificador,
    param_grid,
    cv,
    preprocessor=None,
    return_train_score=False,
    refit_metric="roc_auc",
):
    model = construir_pipeline_modelo_classificacao(classificador, preprocessor)

    grid_search = GridSearchCV(
        model,
        cv=cv,
        param_grid=param_grid,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
        refit=refit_metric,
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def organiza_resultados(resultados):

    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_resultados_expandido
