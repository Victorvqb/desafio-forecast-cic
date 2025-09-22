# src/model_trainer.py (VERSÃO FINAL COM API NATIVA)
"""
Módulo dedicado ao treinamento e otimização do modelo XGBoost,
usando a API nativa para máxima compatibilidade e controle.
"""

import polars as pl
import xgboost as xgb
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import config
import optuna

def tune_hyperparameters(df_modelagem: pl.DataFrame) -> dict:
    """Usa o Optuna com a API nativa do XGBoost para encontrar os melhores hiperparâmetros."""
    X = df_modelagem.select(config.FEATURES).to_pandas()
    y = df_modelagem.select(config.TARGET_COLUMN).to_pandas().squeeze()

    def objective(trial: optuna.Trial) -> float:
        """Função que o Optuna tentará minimizar."""
        # Parâmetros para a API nativa do XGBoost
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'verbosity': 0, # Silencia o output
            'tree_method': 'hist',
            'booster': 'gbtree',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), # 'reg_alpha' é 'alpha' na API nativa
            'lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), # 'reg_lambda' é 'lambda'
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            y_train.clip(0, inplace=True)
            y_test.clip(0, inplace=True)
            y_train_log = np.log1p(y_train)
            y_test_log = np.log1p(y_test)

            # --- MUDANÇA CRÍTICA: Convertendo para o formato DMatrix do XGBoost ---
            dtrain = xgb.DMatrix(X_train, label=y_train_log)
            dvalid = xgb.DMatrix(X_test, label=y_test_log)

            # --- MUDANÇA CRÍTICA: Usando xgb.train com early stopping nativo ---
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=2000,
                evals=[(dvalid, 'validation')],
                early_stopping_rounds=100,
                verbose_eval=False
            )
            
            # A previsão é feita no DMatrix e já usa a melhor iteração
            preds_log = model.predict(dvalid)
            preds = np.expm1(preds_log)
            preds[preds < 0] = 0

            wmape_score = np.sum(np.abs(y_test - preds)) / np.sum(np.abs(y_test))
            scores.append(wmape_score)
            
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)

    print(f"\nMelhor score médio (WMAPE CV): {study.best_value:.6f}")
    print(f"Melhores parâmetros: {study.best_params}")
    return study.best_params

def train_final_model(df_modelagem: pl.DataFrame, best_params: dict):
    """Treina o modelo final com todos os dados e os melhores parâmetros e o salva."""
    X_final = df_modelagem.select(config.FEATURES).to_pandas()
    y_final = df_modelagem.select(config.TARGET_COLUMN).to_pandas().squeeze()
    y_final.clip(0, inplace=True)
    y_final_log = np.log1p(y_final)
    
    # Prepara o DMatrix com todos os dados
    dtrain_final = xgb.DMatrix(X_final, label=y_final_log)
    
    # Junta os parâmetros base com os otimizados
    final_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'mae', 'verbosity': 0,
        'tree_method': 'hist', 'booster': 'gbtree'
    }
    final_params.update(best_params)

    print("\nTreinando modelo final com parâmetros otimizados (API Nativa)...")
    final_model = xgb.train(
        params=final_params,
        dtrain=dtrain_final,
        num_boost_round=2000 # Treina com um número alto de rounds no final
    )

    joblib.dump(final_model, config.PATH_MODELO_FINAL)
    print(f"Modelo final otimizado salvo em: {config.PATH_MODELO_FINAL}")
    return final_model