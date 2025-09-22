# src/feature_engineering.py
"""
Módulo para a criação de todos os atributos (features) para o modelo.
"""
import polars as pl
from datetime import date, timedelta
import config

def create_features(df_semanal: pl.DataFrame) -> pl.DataFrame:
    """Cria um DataFrame de modelagem enriquecido com features preditivas."""
    # --- 1. Feature de Preço ---
    df_com_preco = df_semanal.with_columns(
        (pl.col("net_value_semanal") / pl.when(pl.col("quantidade_semanal") > 0).then(pl.col("quantidade_semanal")).otherwise(1)).alias("preco_medio_semanal")
    )
    df_com_preco = df_com_preco.with_columns(
        pl.col("preco_medio_semanal").shift(1).over(["pdv", "produto"]).alias("preco_lag_1_semana")
    ).with_columns(
        pl.col("preco_lag_1_semana").fill_null(strategy="mean")
    )

    # --- 2. Features de Tempo, Lag e Estatísticas ---
    df_modelagem = df_com_preco.with_columns([
        # MUDANÇA: Novas Features de Calendário Detalhadas
        pl.col("data").dt.day().alias("dia_do_mes"),
        pl.col("data").dt.weekday().alias("dia_da_semana"),
        pl.col("data").dt.week().alias("semana_do_ano"),
        pl.col("data").dt.month().alias("mes"),
        
        # Lags de Vendas
        pl.col("quantidade_semanal").shift(1).over(["pdv", "produto"]).alias("lag_1_semana"),
        pl.col("quantidade_semanal").shift(2).over(["pdv", "produto"]).alias("lag_2_semanas"),
        pl.col("quantidade_semanal").shift(4).over(["pdv", "produto"]).alias("lag_4_semanas"),

        # MUDANÇA: Novas Features Estatísticas de Janela Móvel
        pl.col("quantidade_semanal").shift(1).rolling_mean(window_size=4).over(["pdv", "produto"]).alias("media_movel_4_semanas"),
        pl.col("quantidade_semanal").shift(1).rolling_std(window_size=4).over(["pdv", "produto"]).alias("desvio_padrao_movel_4_semanas"),
        pl.col("quantidade_semanal").shift(1).rolling_min(window_size=4).over(["pdv", "produto"]).alias("min_movel_4_semanas"),
        pl.col("quantidade_semanal").shift(1).rolling_max(window_size=4).over(["pdv", "produto"]).alias("max_movel_4_semanas"),
    ])

    # --- 3. Feature de Feriados ---
    df_modelagem = df_modelagem.with_columns(
        pl.col("data").dt.date().map_elements(
            lambda d: any(abs((d + timedelta(days=i) - f).days) < 1 for f in config.FERIADOS_2022 for i in range(7)),
            return_dtype=pl.Boolean
        ).alias("contem_feriado")
    )
    
    # --- 4. Limpeza Final ---
    df_modelagem = df_modelagem.drop_nulls()

    print("Features de modelagem (com features avançadas) criadas com sucesso.")
    return df_modelagem