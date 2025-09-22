# src/data_processing.py
"""
Módulo responsável pelo carregamento, unificação e processamento inicial dos dados.
"""
import polars as pl
from datetime import date, timedelta
import config

def load_and_join_data() -> pl.DataFrame:
    """Carrega os 3 arquivos parquet e os une em um DataFrame completo."""
    df_pdv = pl.read_parquet(config.PATH_PDV)
    df_transacoes = pl.read_parquet(config.PATH_TRANSACOES)
    df_produtos = pl.read_parquet(config.PATH_PRODUTOS)
    
    df_completo = df_transacoes.join(df_pdv, left_on="internal_store_id", right_on="pdv", how="left")
    df_completo = df_completo.join(df_produtos, left_on="internal_product_id", right_on="produto", how="left")
    
    print("Dados brutos carregados e unidos com sucesso.")
    return df_completo

def clean_and_aggregate(df_completo: pl.DataFrame) -> pl.DataFrame:
    """Limpa, agrega os dados para o nível semanal e corrige o outlier."""
    df_essencial = df_completo.select([
        pl.col("transaction_date").cast(pl.Datetime).alias("data"),
        pl.col("internal_store_id").alias("pdv"),
        pl.col("internal_product_id").alias("produto"),
        pl.col("quantity").alias("quantidade"),
        pl.col("net_value")
    ]).drop_nulls()

    df_semanal = df_essencial.group_by(
        [pl.col("data").dt.truncate("1w"), "pdv", "produto"]
    ).agg([
        pl.sum("quantidade").alias("quantidade_semanal"),
        pl.sum("net_value").alias("net_value_semanal")
    ])

    # Correção do Outlier de Setembro de 2022
    data_pico = date(2022, 9, 5)
    data_anterior = data_pico - timedelta(weeks=1)
    dados_semana_anterior = df_semanal.filter(pl.col("data").dt.date() == data_anterior)
    dados_corrigidos = dados_semana_anterior.with_columns(pl.lit(data_pico).cast(pl.Datetime).alias("data"))
    df_semanal_sem_pico = df_semanal.filter(pl.col("data").dt.date() != data_pico)
    df_semanal_corrigido = pl.concat([df_semanal_sem_pico, dados_corrigidos]).sort("data")
    
    print("Dados agregados para o nível semanal e outlier corrigido.")
    return df_semanal_corrigido