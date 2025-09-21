# src/predict.py
"""
Módulo para gerar as previsões futuras.
A principal função implementa um loop iterativo, onde a previsão de uma semana
é usada para construir as features da semana seguinte.
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date, timedelta
import config

def generate_predictions(model, df_modelagem: pl.DataFrame, df_semanal_corrigido: pl.DataFrame) -> pl.DataFrame:
    """
    Usa um modelo treinado para gerar as previsões para as 5 semanas de Janeiro/2023.

    Args:
        model: O artefato do modelo LightGBM já treinado.
        df_modelagem (pl.DataFrame): Fonte para o histórico recente e entidades únicas.
        df_semanal_corrigido (pl.DataFrame): Fonte para os valores históricos de vendas e preço.
    
    Returns:
        pl.DataFrame: DataFrame formatado para a submissão final.
    """
    print("\nPreparando os DataFrames para Janeiro/2023...")

    # --- 1. Preparar histórico recente e esqueleto futuro ---
    ultimas_semanas_2022_hist = df_modelagem.filter(pl.col("data") > date(2022, 11, 28))
    pdv_produto_unicos = df_modelagem.select(["pdv", "produto"]).unique()
    semanas_jan_2023 = [date(2023, 1, 2), date(2023, 1, 9), date(2023, 1, 16), date(2023, 1, 23), date(2023, 1, 30)]
    df_futuro = pdv_produto_unicos.join(pl.DataFrame({"data": semanas_jan_2023}), how="cross")
    df_futuro = df_futuro.with_columns(pl.col("data").cast(pl.Datetime))

    # --- 2. Alinhar schemas para garantir a concatenação ---
    colunas_que_faltam = set(df_modelagem.columns) - set(df_futuro.columns)
    for col_nome in colunas_que_faltam:
        df_futuro = df_futuro.with_columns(pl.lit(None).alias(col_nome))
    df_futuro = df_futuro.select(df_modelagem.columns)
    df_futuro_com_historia = pl.concat([ultimas_semanas_2022_hist, df_futuro]).sort("data", "pdv", "produto")

    # --- 3. Iniciar loop de previsão iterativa ---
    print("Criando features e prevendo o futuro de forma iterativa...")
    df_previsoes_finais = []
    
    for semana_atual in semanas_jan_2023:
        # Preencher o preço nulo com o último valor conhecido para a semana atual
        df_futuro_com_historia = df_futuro_com_historia.with_columns(
            pl.col("preco_lag_1_semana").fill_null(strategy="forward")
        )
        
        # Criar features para a semana que será prevista
        df_para_prever = df_futuro_com_historia.with_columns([
            pl.col("data").dt.week().alias("semana_do_ano"),
            pl.col("data").dt.month().alias("mes"),
            pl.col("quantidade_semanal").shift(1).over(["pdv", "produto"]).alias("lag_1_semana"),
            pl.col("quantidade_semanal").shift(2).over(["pdv", "produto"]).alias("lag_2_semanas"),
            pl.col("quantidade_semanal").shift(4).over(["pdv", "produto"]).alias("lag_4_semanas"),
            pl.col("quantidade_semanal").shift(1).rolling_mean(window_size=4).over(["pdv", "produto"]).alias("media_movel_4_semanas"),
            pl.lit(False).alias("contem_feriado")
        ]).filter(pl.col("data").dt.date() == semana_atual).fill_null(0)

        # Fazer a previsão para a semana atual
        X_previsao_semana = df_para_prever.select(config.FEATURES).to_pandas()
        previsoes_semana = model.predict(X_previsao_semana)
        previsoes_semana_arr = np.round(previsoes_semana).astype(int)
        previsoes_semana_arr[previsoes_semana_arr < 0] = 0
        
        # Armazenar o resultado da semana
        df_resultado_semana = df_para_prever.select(["data", "pdv", "produto"]).with_columns(
            pl.Series(name="quantidade", values=previsoes_semana_arr)
        )
        df_previsoes_finais.append(df_resultado_semana)

        # Atualizar o histórico com a previsão recém-feita para a próxima iteração
        df_futuro_com_historia = df_futuro_com_historia.update(
            df_resultado_semana.rename({"quantidade": "quantidade_semanal"}),
            on=['data', 'pdv', 'produto']
        )

    # --- 4. Consolidar e formatar a submissão final ---
    print("Formatando o arquivo de submissão...")
    df_submissao = pl.concat(df_previsoes_finais)
    df_submissao = df_submissao.with_columns(
        pl.col("data").dt.week().alias("semana")
    ).select([
        pl.col("semana"),
        pl.col("pdv").cast(pl.Int64),
        pl.col("produto").cast(pl.Int64),
        pl.col("quantidade")
    ])
    
    return df_submissao