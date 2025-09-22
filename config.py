# config.py
"""
Arquivo de configuração central para o projeto de previsão de vendas.
"""
from datetime import date

# --- 1. Caminhos dos Arquivos ---
PATH_PDV = 'data/raw/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet'
PATH_TRANSACOES = 'data/raw/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
PATH_PRODUTOS = 'data/raw/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet'

PATH_OUTPUT = "previsao_final.parquet"
PATH_MODELO_FINAL = "modelo_xgb_final.joblib" # Mudamos o nome para refletir o novo modelo

# --- 2. Parâmetros do Modelo XGBoost ---
XGB_PARAMS = {
    'objective': 'reg:squarederror', # Objetivo padrão, vamos otimizar com Optuna
    'n_estimators': 2000,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist' # Método de construção de árvore rápido
}

# --- 3. Definições de Features e Alvo ---
TARGET_COLUMN = "quantidade_semanal"
FEATURES = [
    'dia_do_mes', 'dia_da_semana', 'semana_do_ano', 'mes',
    'lag_1_semana', 'lag_2_semanas', 'lag_4_semanas',
    'media_movel_4_semanas', 'desvio_padrao_movel_4_semanas',
    'min_movel_4_semanas', 'max_movel_4_semanas',
    'contem_feriado',
    'preco_lag_1_semana'
]

# --- 4. Configurações de Negócio ---
FERIADOS_2022 = [
    date(2022, 4, 21), date(2022, 9, 7), date(2022, 10, 12),
    date(2022, 11, 2), date(2022, 11, 15)
]

# --- 5. Configurações da Otimização (Optuna) ---
OPTUNA_N_TRIALS = 50