# train.py
"""
Script principal para executar o pipeline de treinamento do modelo de previsão.
Passos:
1. Carrega e processa os dados brutos.
2. Executa a engenharia de atributos.
3. Treina um modelo e avalia sua performance em um conjunto de validação.
4. Treina um modelo final com todos os dados e o salva para uso futuro.
"""

from src import data_processing, feature_engineering, model_trainer

def main():
    """Orquestra o pipeline completo de treinamento do modelo."""
    # 1. Carregar e processar os dados
    df_completo = data_processing.load_and_join_data()
    df_semanal_corrigido = data_processing.clean_and_aggregate(df_completo)
    
    # 2. Criar features
    df_modelagem = feature_engineering.create_features(df_semanal_corrigido)
    
    # 3. Treinar e avaliar o modelo para obter métricas de performance
    model_trainer.train_and_evaluate(df_modelagem)
    
    # 4. Treinar e salvar o modelo final com 100% dos dados
    model_trainer.train_final_model(df_modelagem)

if __name__ == "__main__":
    # Garante que o código só será executado quando o script for chamado diretamente
    main()