# generate_submission.py (VERSÃO V5 - Seleção por Relevância Recente)
import joblib
from src import data_processing, feature_engineering, predict
import config
import polars as pl

def main():
    print(f"Carregando modelo treinado de: {config.PATH_MODELO_FINAL}")
    try:
        final_model = joblib.load(config.PATH_MODELO_FINAL)
    except FileNotFoundError:
        print(f"Erro: Modelo não encontrado. Execute 'python train.py' primeiro.")
        return

    print("\nRecarregando e processando dados para consistência...")
    df_completo = data_processing.load_and_join_data()
    df_semanal_corrigido = data_processing.clean_and_aggregate(df_completo)
    df_modelagem = feature_engineering.create_features(df_semanal_corrigido)

    # --- MUDANÇA ESTRATÉGICA: Seleção baseada nas vendas do ÚLTIMO TRIMESTRE ---
    print("\nPriorizando combinações com base na relevância recente (Q4 2022)...")
    vendas_recentes = df_modelagem.filter(pl.col("data").dt.month() >= 10)
    vendas_totais_por_par = vendas_recentes.group_by(["pdv", "produto"]).agg(
        pl.sum(config.TARGET_COLUMN).alias("vendas_totais_q4")
    ).sort("vendas_totais_q4", descending=True)
    
    limite_de_pares = 300000 
    pdv_produto_unicos_priorizados = vendas_totais_por_par.head(limite_de_pares).select(["pdv", "produto"])
    
    print(f"Foram selecionadas as {len(pdv_produto_unicos_priorizados)} combinações mais relevantes recentemente.")

    df_submissao = predict.generate_predictions(final_model, df_modelagem, pdv_produto_unicos_priorizados)
    
    df_submissao.write_parquet(config.PATH_OUTPUT)
    print(f"\n✅ SUCESSO! Arquivo salvo em: {config.PATH_OUTPUT}")
    print(f"Total de linhas: {len(df_submissao)}")
    print(df_submissao.head())

if __name__ == "__main__":
    main()