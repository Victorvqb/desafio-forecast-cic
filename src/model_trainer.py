# src/model_trainer.py
"""
Módulo dedicado ao treinamento, otimização e avaliação do modelo LightGBM.
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
import optuna

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

def tune_hyperparameters(df_modelagem: pl.DataFrame) -> dict:
    """
    Usa o Optuna para encontrar a melhor combinação de hiperparâmetros para o modelo.

    Args:
        df_modelagem (pl.DataFrame): O DataFrame completo com todas as features.

    Returns:
        dict: Um dicionário contendo os melhores parâmetros encontrados.
    """
    # 1. Divisão de dados para a otimização
    data_de_corte = df_modelagem.get_column("data").max() - timedelta(weeks=4)
    treino = df_modelagem.filter(pl.col("data") < data_de_corte)
    validacao = df_modelagem.filter(pl.col("data") >= data_de_corte)

    X_treino = treino.select(config.FEATURES).to_pandas()
    y_treino = treino.select(config.TARGET_COLUMN).to_pandas().squeeze()
    X_valid = validacao.select(config.FEATURES).to_pandas()
    y_valid = validacao.select(config.TARGET_COLUMN).to_pandas().squeeze()

    def objective(trial: optuna.Trial) -> float:
        """
        Função que o Optuna tentará minimizar.
        Para cada 'trial', ele sugere um conjunto de parâmetros, treina um modelo
        e retorna a pontuação de WMAPE na validação.
        """
        # 2. Definição do espaço de busca dos parâmetros
        params = {
            'objective': wmape_objective,
            'metric': 'mae',
            'n_estimators': 2000,
            'random_state': 42,
            'n_jobs': -1,
            # Parâmetros que o Optuna irá otimizar
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
        }

        # 3. Treinamento e avaliação do trial
        model = lgb.LGBMRegressor(**params)
        model.fit(X_treino, y_treino,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric=wmape_eval,
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # 4. Cálculo da métrica a ser otimizada
        preds = model.predict(X_valid)
        wmape_score = np.sum(np.abs(y_valid - preds)) / np.sum(np.abs(y_valid))
        return wmape_score

    # 5. Execução do estudo de otimização
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)

    print("\nOtimização de Hiperparâmetros concluída!")
    print(f"Melhor pontuação (WMAPE): {study.best_value:.6f}")
    print("Melhores parâmetros encontrados:")
    print(study.best_params)

    return study.best_params


def train_final_model(df_modelagem: pl.DataFrame, best_params: dict):
    """
    Treina o modelo final com todos os dados e os melhores parâmetros e o salva.
    
    Args:
        df_modelagem (pl.DataFrame): O DataFrame completo com todas as features.
        best_params (dict): Dicionário com os melhores hiperparâmetros encontrados pelo Optuna.

    Returns:
        O objeto do modelo LightGBM treinado.
    """
    X_final = df_modelagem.select(config.FEATURES).to_pandas()
    y_final = df_modelagem.select(config.TARGET_COLUMN).to_pandas().squeeze()
    
    # Combina os parâmetros fixos com os melhores encontrados pelo Optuna
    final_params = config.LGBM_PARAMS.copy()
    final_params['objective'] = wmape_objective
    final_params.update(best_params) # Atualiza com os melhores parâmetros!

    final_model = lgb.LGBMRegressor(**final_params)
    print("\nTreinando modelo final com todos os dados e parâmetros otimizados...")
    final_model.fit(X_final, y_final)

    # Salva o modelo treinado em disco para uso posterior pelo script de previsão
    joblib.dump(final_model, config.PATH_MODELO_FINAL)
    print(f"Modelo final otimizado salvo em: {config.PATH_MODELO_FINAL}")
    
    return final_model