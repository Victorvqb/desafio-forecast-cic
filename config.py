# config.py
"""
Arquivo de configuração central para o projeto de previsão de vendas.

Armazena todos os caminhos de arquivos, parâmetros de modelo e listas de features
para facilitar a manutenção e a reprodutibilidade.
"""

from datetime import date

# --- 1. Caminhos dos Arquivos (Relativos à raiz do projeto) ---
# Esta estrutura assume que os scripts são executados da pasta raiz do projeto.
PATH_PDV = 'data/raw/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet'
PATH_TRANSACOES = 'data/raw/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
PATH_PRODUTOS = 'data/raw/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet'

# Local onde os artefatos gerados serão salvos
PATH_OUTPUT = "previsao_final.parquet" # Arquivo final de previsões
PATH_MODELO_FINAL = "modelo_lgbm_final.joblib" # .joblib é o formato recomendado para salvar modelos sklearn/lgbm


# --- 2. Parâmetros do Modelo LightGBM ---
# Estes são os parâmetros base. A otimização do Optuna irá sobrescrevê-los.
LGBM_PARAMS = {
    'objective': None,          # Será definido dinamicamente no script de treino
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8
}

# Parâmetros para o método .fit() do modelo
LGBM_FIT_PARAMS = {
    'eval_metric': None,        # Será definido dinamicamente no script de treino
    'callbacks': []             # Será definido dinamicamente no script de treino
}


# --- 3. Definições de Features e Alvo ---
TARGET_COLUMN = "quantidade_semanal"
FEATURES = [
    'semana_do_ano', 'mes',
    'lag_1_semana', 'lag_2_semanas', 'lag_4_semanas',
    'media_movel_4_semanas',
    'contem_feriado',
    'preco_lag_1_semana'
]


# --- 4. Configurações de Negócio ---
# Lista de feriados nacionais de 2022 para criação da feature correspondente
FERIADOS_2022 = [
    date(2022, 4, 21), date(2022, 9, 7), date(2022, 10, 12),
    date(2022, 11, 2), date(2022, 11, 15)
]

# --- 5. Configurações da Otimização (Optuna) ---
OPTUNA_N_TRIALS = 50 # Número de tentativas que o Optuna fará. Aumente para uma busca mais exaustiva.