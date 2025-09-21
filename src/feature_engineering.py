# src/feature_engineering.py
"""
Módulo para a criação de todos os atributos (features) que serão usados
pelo modelo de machine learning. Transforma os dados processados em um
formato rico e informativo para a modelagem.
"""

import polars as pl
from datetime import date, timedelta
import config

def create_features(df_semanal: pl.DataFrame) -> pl.DataFrame:
    """
    Cria um DataFrame de modelagem enriquecido com features preditivas.

    Args:
        df_semanal (pl.DataFrame): DataFrame com dados semanais limpos.

    Returns:
        pl.DataFrame: DataFrame pronto para ser usado no treinamento do modelo.
    """
    # --- 1. Feature de Preço ---
    # Calcula o preço médio semanal e cria um lag de 1 semana para usar como feature.
    # Nulos (gerados no início da série de cada produto) são preenchidos com a média geral.
    df_com_preco = df_semanal.with_columns(
        (pl.col("net_value_semanal") / pl.when(pl.col("quantidade_semanal") > 0).then(pl.col("quantidade_semanal")).otherwise(1)).alias("preco_medio_semanal")
    )
    df_com_preco = df_com_preco.with_columns(
        pl.col("preco_medio_semanal").shift(1).over(["pdv", "produto"]).alias("preco_lag_1_semana")
    ).with_columns(
        pl.col("preco_lag_1_semana").fill_null(strategy="mean")
    )

    # --- 2. Features de Tempo, Lag e Janela Móvel ---
    # Cria atributos baseados no tempo (mês, semana) e no histórico de vendas (lags e médias).
    # O .over(["pdv", "produto"]) é crucial: garante que os cálculos de lag e média móvel
    # sejam feitos para cada série temporal (cada combinação de loja/produto) individualmente.
    df_modelagem = df_com_preco.with_columns([
        pl.col("data").dt.week().alias("semana_do_ano"),
        pl.col("data").dt.month().alias("mes"),
        pl.col("quantidade_semanal").shift(1).over(["pdv", "produto"]).alias("lag_1_semana"),
        pl.col("quantidade_semanal").shift(2).over(["pdv", "produto"]).alias("lag_2_semanas"),
        pl.col("quantidade_semanal").shift(4).over(["pdv", "produto"]).alias("lag_4_semanas"),
        pl.col("quantidade_semanal").shift(1).rolling_mean(window_size=4).over(["pdv", "produto"]).alias("media_movel_4_semanas")
    ])

    # --- 3. Feature de Feriados ---
    # Adiciona uma flag booleana (True/False) se a semana de venda contém um feriado.
    df_modelagem = df_modelagem.with_columns(
        pl.col("data").dt.date().map_elements(
            lambda d: any(abs((d + timedelta(days=i) - f).days) < 1 for f in config.FERIADOS_2022 for i in range(7)),
            return_dtype=pl.Boolean
        ).alias("contem_feriado")
    )

    # --- 4. Limpeza Final ---
    # Remove as primeiras linhas de cada série que contêm nulos devido à criação dos lags.
    df_modelagem = df_modelagem.drop_nulls()

    print("Features de modelagem criadas com sucesso.")
    return df_modelagem