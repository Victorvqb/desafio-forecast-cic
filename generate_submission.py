# generate_submission.py (VERSÃO FINAL COM PRIORITIZAÇÃO)

import joblib
from src import data_processing, feature_engineering, predict
import config
import polars as pl

def main():
    """
    Orquestra o pipeline de geração do arquivo de submissão, com priorização
    para respeitar o limite de 1.5 milhão de linhas da plataforma.
    """
    # Etapa 1: Carregar o modelo final treinado
    print(f"Carregando modelo treinado de: {config.PATH_MODELO_FINAL}")
    try:
        final_model = joblib.load(config.PATH_MODELO_FINAL)
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado em '{config.PATH_MODELO_FINAL}'")
        print("Por favor, execute 'python train.py' primeiro para treinar e salvar o modelo.")
        return

    # Etapa 2: Recarregar e processar dados
    print("\nRecarregando e processando dados para consistência...")
    df_completo = data_processing.load_and_join_data()
    df_semanal_corrigido = data_processing.clean_and_aggregate(df_completo)
    df_modelagem = feature_engineering.create_features(df_semanal_corrigido)


    print("\nPriorizando as combinações de PDV/Produto mais relevantes...")
    
    # 1. Calcular o total de vendas por par pdv/produto em 2022
    vendas_totais_por_par = df_modelagem.group_by(["pdv", "produto"]).agg(
        pl.sum(config.TARGET_COLUMN).alias("vendas_totais_2022")
    ).sort("vendas_totais_2022", descending=True)

    # 2. Definir o limite e selecionar os top pares
    # O limite é 1.500.000 linhas / 5 semanas = 300.000 pares
    limite_de_pares = 300000 
    pdv_produto_unicos_priorizados = vendas_totais_por_par.head(limite_de_pares).select(["pdv", "produto"])
    
    print(f"Foram selecionadas as {len(pdv_produto_unicos_priorizados)} combinações mais vendidas para a previsão.")
    print(f"Total de linhas a serem geradas: {len(pdv_produto_unicos_priorizados) * 5}")
    
    # Etapa 3: Gerar as previsões (agora usando a lista priorizada)
    df_submissao = predict.generate_predictions(
        model=final_model, 
        df_modelagem=df_modelagem, 
        pdv_produto_unicos=pdv_produto_unicos_priorizados # Passando a lista filtrada como argumento
    )

    # Etapa 4: Salvar o arquivo final no formato Parquet (ou CSV, se preferir)
    df_submissao.write_parquet(config.PATH_OUTPUT)
    
    print(f"\n✅ SUCESSO! Seu arquivo de previsão foi salvo em: {config.PATH_OUTPUT}")
    print(f"Total de linhas no arquivo final: {len(df_submissao)}")
    print("Amostra do arquivo de submissão:")
    print(df_submissao.head())

if __name__ == "__main__":
    main()