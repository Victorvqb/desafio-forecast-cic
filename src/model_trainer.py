# src/model_trainer.py (VERSÃO V5 - Foco em LightGBM e Validação Simples)
import polars as pl
import lightgbm as lgb
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import joblib
import config
import optuna
from tqdm import tqdm # Para a barra de progresso

def tune_hyperparameters(df_modelagem: pl.DataFrame) -> dict:
    data_de_corte = df_modelagem.get_column("data").max() - timedelta(weeks=4)
    treino = df_modelagem.filter(pl.col("data") < data_de_corte)
    validacao = df_modelagem.filter(pl.col("data") >= data_de_corte)
    X_treino = treino.select(config.FEATURES).to_pandas()
    y_treino = treino.select(config.TARGET_COLUMN).to_pandas().squeeze()
    X_valid = validacao.select(config.FEATURES).to_pandas()
    y_valid = validacao.select(config.TARGET_COLUMN).to_pandas().squeeze()

    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
            'random_state': 42, 'n_jobs': -1, 'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'num_leaves': trial.suggest_int('num_leaves', 10, 40),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        y_train_log = np.log1p(y_train.clip(0))
        model = lgb.LGBMRegressor(**params)
        model.fit(X_treino, y_train_log, eval_set=[(X_valid, np.log1p(y_valid.clip(0)))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        preds_log = model.predict(X_valid)
        preds = np.expm1(preds_log)
        preds[preds < 0] = 0
        wmape_score = np.sum(np.abs(y_valid - preds)) / np.sum(np.abs(y_valid))
        return wmape_score

    pbar = tqdm(total=config.OPTUNA_N_TRIALS, desc="Otimizando Hiperparâmetros")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: (pbar.update(1), objective(trial))[1], n_trials=config.OPTUNA_N_TRIALS)
    pbar.close()

    print(f"\nMelhor score (WMAPE Validação): {study.best_value:.6f}")
    print(f"Melhores parâmetros: {study.best_params}")
    return study.best_params

def train_final_model(df_modelagem: pl.DataFrame, best_params: dict):
    # Lógica para treinar o modelo final com LightGBM
    X_final = df_modelagem.select(config.FEATURES).to_pandas()
    y_final = df_modelagem.select(config.TARGET_COLUMN).to_pandas().squeeze()
    y_final_log = np.log1p(y_final.clip(0))
    final_params = config.LGBM_PARAMS.copy()
    final_params['objective'] = 'regression_l1'
    final_params.update(best_params)
    final_model = lgb.LGBMRegressor(**final_params)
    print("\nTreinando modelo final com parâmetros otimizados (em escala log)...")
    final_model.fit(X_final, y_final_log)
    joblib.dump(final_model, config.PATH_MODELO_FINAL)
    print(f"Modelo final otimizado salvo em: {config.PATH_MODELO_FINAL}")
    return final_model