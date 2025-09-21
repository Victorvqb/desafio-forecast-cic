# src/model_trainer.py
"""
Módulo dedicado ao treinamento e avaliação do modelo LightGBM.
Inclui as funções customizadas para a otimização de WMAPE e as rotinas
para treinamento de validação e treinamento final.
"""

import polars as pl
import lightgbm as lgb
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import joblib
import config

def wmape_objective(y_true, y_pred):
    """
    Função objetivo customizada para o LightGBM.
    Ela calcula o gradiente e o hessiano para a perda L1 (Erro Absoluto),
    que direciona o modelo a minimizar o numerador da métrica WMAPE.
    """
    grad = np.sign(y_pred - y_true)
    hess = np.ones_like(y_true)
    return grad, hess

def wmape_eval(y_true, y_pred):
    """
    Métrica de avaliação customizada para o LightGBM.
    Calcula o WMAPE real para ser monitorado durante o treinamento
    e usado para o 'early stopping'.
    """
    wmape_score = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    # Formato de retorno: (nome_da_metrica, score, menor_e_melhor)
    return "wmape", wmape_score, False

def train_and_evaluate(df_modelagem: pl.DataFrame):
    """
    Executa um ciclo completo de treinamento e validação.
    Separa os dados (usando Dezembro/2022 para validação), treina o modelo
    e imprime as métricas de performance para avaliação.

    Args:
        df_modelagem (pl.DataFrame): O DataFrame completo com todas as features.
    """
    # 1. Divisão de dados temporal
    data_de_corte = df_modelagem.get_column("data").max() - timedelta(weeks=4)
    treino = df_modelagem.filter(pl.col("data") < data_de_corte)
    validacao = df_modelagem.filter(pl.col("data") >= data_de_corte)

    # 2. Separação de features (X) e alvo (y)
    X_treino = treino.select(config.FEATURES).to_pandas()
    y_treino = treino.select(config.TARGET_COLUMN).to_pandas().squeeze()
    X_valid = validacao.select(config.FEATURES).to_pandas()
    y_valid = validacao.select(config.TARGET_COLUMN).to_pandas().squeeze()

    print("\nIniciando o treinamento do modelo para validação...")
    
    # 3. Configuração e treinamento
    model_params = config.LGBM_PARAMS
    model_params['objective'] = wmape_objective
    
    fit_params = config.LGBM_FIT_PARAMS
    fit_params['eval_metric'] = wmape_eval
    fit_params['callbacks'] = [lgb.early_stopping(100, verbose=True)]

    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_treino, y_treino, eval_set=[(X_valid, y_valid)], **fit_params)

    # 4. Avaliação de performance
    print("\n--- PERFORMANCE DO MODELO (VALIDAÇÃO) ---")
    previsoes = model.predict(X_valid)
    previsoes_arredondadas = np.round(previsoes)
    
    mae = mean_absolute_error(y_valid, previsoes_arredondadas)
    wmape = np.sum(np.abs(y_valid - previsoes_arredondadas)) / np.sum(np.abs(y_valid))

    print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
    print(f"WMAPE na validação: {wmape:.6f}")
    
def train_final_model(df_modelagem: pl.DataFrame):
    """
    Treina o modelo final usando 100% dos dados de 2022 e o salva em disco.

    Args:
        df_modelagem (pl.DataFrame): O DataFrame completo com todas as features.
    
    Returns:
        O objeto do modelo LightGBM treinado.
    """
    X_final = df_modelagem.select(config.FEATURES).to_pandas()
    y_final = df_modelagem.select(config.TARGET_COLUMN).to_pandas().squeeze()

    model_params = config.LGBM_PARAMS
    model_params['objective'] = wmape_objective
    
    final_model = lgb.LGBMRegressor(**model_params)
    print("\nTreinando modelo final com todos os dados...")
    final_model.fit(X_final, y_final)
    
    # Salva o modelo treinado em disco para uso posterior pelo script de previsão
    joblib.dump(final_model, config.PATH_MODELO_FINAL)
    print(f"Modelo final treinado e salvo em: {config.PATH_MODELO_FINAL}")
    
    return final_model