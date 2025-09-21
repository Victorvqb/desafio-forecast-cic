# src/data_processing.py
"""
Módulo responsável pelo carregamento, unificação e processamento inicial dos dados.
As funções aqui preparam o terreno para a engenharia de atributos.
"""

import polars as pl
from datetime import date, timedelta
import config

def load_and_join_data() -> pl.DataFrame:
    """
    Carrega os três arquivos parquet de dados brutos (PDVs, Transações, Produtos)
    e os une em um único DataFrame Polars.

    A união é feita com um 'left join' a partir da tabela de transações para garantir
    que nenhuma venda seja perdida, mesmo que um PDV ou produto não esteja no cadastro.

    Returns:
        pl.DataFrame: DataFrame unificado contendo todos os dados brutos.
    """
    # Carrega os arquivos usando os caminhos definidos no config
    df_pdv = pl.read_parquet(config.PATH_PDV)
    df_transacoes = pl.read_parquet(config.PATH_TRANSACOES)
    df_produtos = pl.read_parquet(config.PATH_PRODUTOS)

    # Une transações com o cadastro de PDVs
    df_completo = df_transacoes.join(df_pdv, left_on="internal_store_id", right_on="pdv", how="left")
    # Une o resultado com o cadastro de produtos
    df_completo = df_completo.join(df_produtos, left_on="internal_product_id", right_on="produto", how="left")

    print("Dados brutos carregados e unidos com sucesso.")
    return df_completo

def clean_and_aggregate(df_completo: pl.DataFrame) -> pl.DataFrame:
    """
    Realiza a limpeza e agregação principal dos dados.
    1. Seleciona colunas essenciais e remove registros incompletos.
    2. Agrega os dados transacionais para o nível semanal por PDV e produto.
    3. Identifica e corrige o outlier de vendas de Setembro/2022, substituindo
       os dados da semana anômala pelos da semana anterior para manter a consistência.

    Args:
        df_completo (pl.DataFrame): O DataFrame unificado de dados brutos.

    Returns:
        pl.DataFrame: DataFrame limpo e agregado no nível semanal, pronto para feature engineering.
    """
    # 1. Seleção inicial e conversão de tipos
    df_essencial = df_completo.select([
        pl.col("transaction_date").cast(pl.Datetime).alias("data"),
        pl.col("internal_store_id").alias("pdv"),
        pl.col("internal_product_id").alias("produto"),
        pl.col("quantity").alias("quantidade"),
        pl.col("net_value")
    ]).drop_nulls()

    # 2. Agregação para o nível semanal
    df_semanal = df_essencial.group_by(
        [pl.col("data").dt.truncate("1w"), "pdv", "produto"]
    ).agg([
        pl.sum("quantidade").alias("quantidade_semanal"),
        pl.sum("net_value").alias("net_value_semanal")
    ])

    # 3. Tratamento do Outlier de Set/2022 (substituição pela semana anterior)
    data_pico = date(2022, 9, 5)
    data_anterior = data_pico - timedelta(weeks=1)
    dados_semana_anterior = df_semanal.filter(pl.col("data").dt.date() == data_anterior)
    dados_corrigidos = dados_semana_anterior.with_columns(pl.lit(data_pico).cast(pl.Datetime).alias("data"))
    df_semanal_sem_pico = df_semanal.filter(pl.col("data").dt.date() != data_pico)
    df_semanal_corrigido = pl.concat([df_semanal_sem_pico, dados_corrigidos]).sort("data")

    print("Dados agregados para o nível semanal e outlier corrigido.")
    return df_semanal_corrigido