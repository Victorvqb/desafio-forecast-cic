# train.py
"""
Script principal para executar o pipeline de treinamento do modelo de previsão.
Passos:
1. Carrega e processa os dados brutos.
2. Executa a engenharia de atributos.
3. Utiliza o Optuna para encontrar os melhores hiperparâmetros.
4. Treina um modelo final com os parâmetros otimizados e o salva para uso futuro.
"""

from src import data_processing, feature_engineering, model_trainer

def main():
    """Orquestra o pipeline completo de treinamento do modelo."""
    # Etapa 1: Carregar e processar os dados
    df_completo = data_processing.load_and_join_data()
    df_semanal_corrigido = data_processing.clean_and_aggregate(df_completo)
    
    # Etapa 2: Criar features
    df_modelagem = feature_engineering.create_features(df_semanal_corrigido)
    
    # Etapa 3: Otimizar os hiperparâmetros com Optuna
    best_params = model_trainer.tune_hyperparameters(df_modelagem)
    
    # Etapa 4: Treinar e salvar o modelo final com os melhores parâmetros
    model_trainer.train_final_model(df_modelagem, best_params)
    
    print("\nPipeline de treinamento concluído com sucesso!")

if __name__ == "__main__":
    # Garante que o código só será executado quando o script for chamado diretamente
    main()