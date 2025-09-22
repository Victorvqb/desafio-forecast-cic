# src/predict.py
"""
Módulo para gerar as previsões futuras usando um modelo Booster nativo do XGBoost.

A principal função, 'generate_predictions', implementa um loop iterativo.
Neste loop, a previsão de uma semana é usada para construir as features da
semana seguinte, tornando o processo de previsão para múltiplos passos no
futuro mais robusto e preciso.
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date, timedelta
import config
import xgboost as xgb

def generate_predictions(model, df_modelagem: pl.DataFrame, pdv_produto_unicos: pl.DataFrame) -> pl.DataFrame:
    """
    Usa um modelo Booster treinado para gerar as previsões para as 5 semanas de Janeiro/2023.

    Args:
        model: O artefato do modelo XGBoost já treinado (objeto Booster).
        df_modelagem (pl.DataFrame): O DataFrame completo com features, usado como
                                     fonte para o histórico recente e para a lista de produtos.
        pdv_produto_unicos (pl.DataFrame): DataFrame contendo os pares PDV/Produto prioritários
                                             para os quais a previsão deve ser gerada.

    Returns:
        pl.DataFrame: Um DataFrame final formatado para a submissão, contendo as
                      colunas 'semana', 'pdv', 'produto', 'quantidade'.
    """
    print("\nPreparando os DataFrames para Janeiro/2023...")

    # --- 1. Preparar histórico recente e esqueleto do futuro ---
    # Filtra as últimas semanas de 2022 para "semear" as features de lag para janeiro.
    ultimas_semanas_2022_hist = df_modelagem.filter(pl.col("data") > date(2022, 11, 28))

    # Define as 5 semanas de janeiro para as quais faremos a previsão.
    semanas_jan_2023 = [date(2023, 1, 2), date(2023, 1, 9), date(2023, 1, 16), date(2023, 1, 23), date(2023, 1, 30)]

    # Cria o "esqueleto" de dados futuros, com todas as combinações de PDV/Produto para cada semana de janeiro.
    df_futuro = pdv_produto_unicos.join(pl.DataFrame({"data": semanas_jan_2023}), how="cross")
    df_futuro = df_futuro.with_columns(pl.col("data").cast(pl.Datetime))

    # --- 2. Alinhar schemas para garantir a concatenação ---
    # Adiciona as colunas que faltam ao df_futuro com valores nulos para que ele tenha a mesma
    # estrutura do DataFrame histórico.
    colunas_que_faltam = set(df_modelagem.columns) - set(df_futuro.columns)
    for col_nome in colunas_que_faltam:
        df_futuro = df_futuro.with_columns(pl.lit(None).alias(col_nome))
    
    # Força a mesma ordem de colunas do histórico para evitar erros.
    df_futuro = df_futuro.select(df_modelagem.columns)

    # Concatena o histórico recente com o esqueleto do futuro para criar uma única timeline.
    df_futuro_com_historia = pl.concat([
        ultimas_semanas_2022_hist,
        df_futuro
    ]).sort("data", "pdv", "produto")

    # --- 3. Iniciar loop de previsão iterativa ---
    print("Criando features e prevendo o futuro de forma iterativa...")
    df_previsoes_finais = []

    for semana_atual in semanas_jan_2023:
        # Para cada semana do futuro, preenchemos os valores de preço nulos com o último valor conhecido.
        df_futuro_com_historia = df_futuro_com_historia.with_columns(
            pl.col("preco_lag_1_semana").fill_null(strategy="forward")
        )

        # Criamos o conjunto completo de features para a semana que queremos prever.
        df_para_prever = df_futuro_com_historia.with_columns([
            pl.col("data").dt.week().alias("semana_do_ano"),
            pl.col("data").dt.month().alias("mes"),
            pl.col("quantidade_semanal").shift(1).over(["pdv", "produto"]).alias("lag_1_semana"),
            pl.col("quantidade_semanal").shift(2).over(["pdv", "produto"]).alias("lag_2_semanas"),
            pl.col("quantidade_semanal").shift(4).over(["pdv", "produto"]).alias("lag_4_semanas"),
            pl.col("quantidade_semanal").shift(1).rolling_mean(window_size=4).over(["pdv", "produto"]).alias("media_movel_4_semanas"),
            pl.col("quantidade_semanal").shift(1).rolling_std(window_size=4).over(["pdv", "produto"]).alias("desvio_padrao_movel_4_semanas"),
            pl.col("quantidade_semanal").shift(1).rolling_min(window_size=4).over(["pdv", "produto"]).alias("min_movel_4_semanas"),
            pl.col("quantidade_semanal").shift(1).rolling_max(window_size=4).over(["pdv", "produto"]).alias("max_movel_4_semanas"),
            pl.lit(False).alias("contem_feriado")
        ]).filter(
            pl.col("data").dt.date() == semana_atual
        ).fill_null(0) # Substituímos nulos restantes por 0 antes de prever.

        # Prepara os dados para o formato do modelo (Pandas)
        X_previsao_semana = df_para_prever.select(config.FEATURES).to_pandas()

        # Converte para o formato DMatrix nativo do XGBoost
        dpredict = xgb.DMatrix(X_previsao_semana)
        # O modelo prevê em escala logarítmica
        previsoes_log = model.predict(dpredict)

        # CRÍTICO: Aplica a transformação inversa (exponencial) para voltar à escala original de unidades.
        previsoes_original = np.expm1(previsoes_log)

        # Arredonda para o inteiro mais próximo e garante que não haja previsões negativas.
        previsoes_arr = np.round(previsoes_original).astype(int)
        previsoes_arr[previsoes_arr < 0] = 0

        # Armazena o resultado da semana em um DataFrame formatado.
        df_resultado_semana = df_para_prever.select(["data", "pdv", "produto"]).with_columns(
            pl.Series(name="quantidade", values=previsoes_arr)
        )
        df_previsoes_finais.append(df_resultado_semana)

        # ATUALIZA o histórico com a previsão recém-feita para que ela possa ser usada
        # como lag para a próxima iteração do loop.
        df_futuro_com_historia = df_futuro_com_historia.update(
            df_resultado_semana.rename({"quantidade": "quantidade_semanal"}),
            on=['data', 'pdv', 'produto']
        )

    # --- 4. Consolidar e formatar a submissão final ---
    print("Formatando o arquivo de submissão...")
    # Concatena os resultados das 5 semanas previstas
    df_submissao = pl.concat(df_previsoes_finais)
    
    # Formata as colunas para o padrão de entrega exigido pelo desafio
    df_submissao = df_submissao.with_columns(
        pl.col("data").dt.week().alias("semana")
    ).select([
        pl.col("semana"),
        pl.col("pdv").cast(pl.Int64),
        pl.col("produto").cast(pl.Int64),
        pl.col("quantidade")
    ])

    return df_submissao