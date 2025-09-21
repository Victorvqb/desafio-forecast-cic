# generate_submission.py
"""
Script principal para gerar o arquivo de submissão final.
Este script assume que um modelo já foi treinado e salvo pelo 'train.py'.
Passos:
1. Carrega o artefato do modelo treinado.
2. Recarrega e processa os dados históricos para garantir consistência.
3. Chama a função de previsão para gerar os dados de Jan/2023.
4. Salva o resultado no formato CSV especificado.
"""
import joblib
from src import data_processing, feature_engineering, predict
import config
import polars as pl

def main():
    """Orquestra o pipeline de geração do arquivo de submissão."""
    # 1. Carregar o modelo final treinado do disco
    print(f"Carregando modelo treinado de: {config.PATH_MODELO_FINAL}")
    try:
        final_model = joblib.load(config.PATH_MODELO_FINAL)
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado em '{config.PATH_MODELO_FINAL}'")
        print("Por favor, execute 'python train.py' primeiro para treinar e salvar o modelo.")
        return

    # 2. Recarregar e processar os dados históricos.
    #    Isso garante que o script de previsão possa ser executado de forma independente.
    print("\nRecarregando e processando dados para consistência...")
    df_completo = data_processing.load_and_join_data()
    df_semanal_corrigido = data_processing.clean_and_aggregate(df_completo)
    df_modelagem = feature_engineering.create_features(df_semanal_corrigido)

    # 3. Gerar as previsões usando a lógica do módulo 'predict'
    df_submissao = predict.generate_predictions(final_model, df_modelagem, df_semanal_corrigido)

    # 4. Salvar o arquivo final
    df_submissao.write_csv(config.PATH_OUTPUT, separator=';')
    
    print(f"\n✅ SUCESSO! Seu arquivo de previsão foi salvo em: {config.PATH_OUTPUT}")
    print("Amostra do arquivo de submissão:")
    print(df_submissao.head())

if __name__ == "__main__":
    main()