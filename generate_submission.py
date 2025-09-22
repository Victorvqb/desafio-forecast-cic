# generate_submission.py
"""
Script principal para gerar o arquivo de submissão final.
Este script assume que um modelo já foi treinado e salvo pelo 'train.py'.
Passos:
1. Carrega o artefato do modelo treinado.
2. Recarrega e processa os dados históricos para garantir consistência.
3. Seleciona os PDVs/Produtos mais relevantes para respeitar o limite de linhas.
4. Chama a função de previsão para gerar os dados de Jan/2023.
5. Salva o resultado no formato Parquet.
"""
import joblib
from src import data_processing, feature_engineering, predict
import config
import polars as pl

def main():
    """Orquestra o pipeline de geração do arquivo de submissão."""
    # Etapa 1: Carregar o modelo final treinado do disco
    print(f"Carregando modelo treinado de: {config.PATH_MODELO_FINAL}")
    try:
        final_model = joblib.load(config.PATH_MODELO_FINAL)
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado em '{config.PATH_MODELO_FINAL}'")
        print("Por favor, execute 'python train.py' primeiro para treinar e salvar o modelo.")
        return

    # Etapa 2: Recarregar e processar os dados históricos.
    print("\nRecarregando e processando dados para consistência...")
    df_completo = data_processing.load_and_join_data()
    df_semanal_corrigido = data_processing.clean_and_aggregate(df_completo)
    df_modelagem = feature_engineering.create_features(df_semanal_corrigido)

    # Etapa 3: Priorizar as combinações de PDV/Produto mais relevantes
    print("\nPriorizando as combinações de PDV/Produto mais relevantes...")
    vendas_totais_por_par = df_modelagem.group_by(["pdv", "produto"]).agg(
        pl.sum(config.TARGET_COLUMN).alias("vendas_totais_2022")
    ).sort("vendas_totais_2022", descending=True)
    
    limite_de_pares = 300000 
    pdv_produto_unicos_priorizados = vendas_totais_por_par.head(limite_de_pares).select(["pdv", "produto"])
    
    print(f"Foram selecionadas as {len(pdv_produto_unicos_priorizados)} combinações mais vendidas para a previsão.")

    # Etapa 4: Gerar as previsões usando a lógica do módulo 'predict'
    df_submissao = predict.generate_predictions(final_model, df_modelagem, pdv_produto_unicos_priorizados)

    # Etapa 5: Salvar o arquivo final
    df_submissao.write_parquet(config.PATH_OUTPUT)
    
    print(f"\n✅ SUCESSO! Seu arquivo de previsão foi salvo em: {config.PATH_OUTPUT}")
    print(f"Total de linhas no arquivo final: {len(df_submissao)}")
    print("Amostra do arquivo de submissão:")
    print(df_submissao.head())

if __name__ == "__main__":
    main()