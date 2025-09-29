<h1 align="center">
  Modelo Preditivo de Vendas para Varejo (Forecast)
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/status-finalizado-brightgreen" alt="Status: Finalizado">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python: 3.9+">
  <img src="https://img.shields.io/badge/modelo-LightGBM_/_XGBoost-purple.svg" alt="Modelo: LightGBM / XGBoost">
  <img src="https://img.shields.io/badge/licença-MIT-lightgrey.svg" alt="Licença: MIT">
</p>

## 📖 Resumo do Projeto

Este repositório contém uma solução de Machine Learning de ponta a ponta para um desafio de previsão de vendas (forecast). O projeto utiliza um histórico de vendas de 2022 para prever a demanda semanal por Ponto de Venda (PDV) e SKU (produto) para as cinco primeiras semanas de Janeiro de 2023.

O código foi inteiramente modularizado seguindo as melhores práticas de engenharia de software, garantindo que a solução seja limpa, reprodutível e de fácil manutenção.

## 👨‍💻 Autor

Este projeto foi desenvolvido por:

* **Victor Barbosa** 
    * LinkedIn: https://www.linkedin.com/in/viictorbarbosaa/

## 🎯 Objetivo de Negócio

Apoiar a operação de varejo na reposição inteligente de produtos, otimizando a gestão de estoques e evitando rupturas de gôndola através de uma previsão de demanda precisa e automatizada.

## 🛠️ Principais Bibliotecas e Ferramentas

* **Manipulação de Dados:** Polars (para alta performance em grandes datasets)
* **Modelagem de Machine Learning:** LightGBM e XGBoost
* **Otimização de Hiperparâmetros:** Optuna
* **Estrutura e Avaliação:** Scikit-learn, NumPy
* **Visualização:** Matplotlib

<br>

<details>
  <summary><strong>📁 Clique para ver a Estrutura do Projeto</strong></summary>

  ```
  ├── config.py                   # Arquivo de configuração central
  ├── data/
  │   └── raw/                    # Pasta para os dados brutos .parquet (a ser criada)
  ├── src/
  │   ├── data_processing.py      # Módulo de carregamento e limpeza de dados
  │   ├── feature_engineering.py  # Módulo de criação de atributos (features)
  │   ├── model_trainer.py        # Módulo de treinamento e otimização do modelo
  │   └── predict.py              # Módulo com a lógica de previsão iterativa
  ├── train.py                    # Script principal para TREINAR o modelo
  ├── generate_submission.py      # Script principal para GERAR o arquivo de previsão
  ├── requirements.txt            # Lista de dependências do projeto
  └── modelo_lgbm_final.joblib    # Artefato do modelo treinado (gerado por train.py)
  ```
</details>

## 🚀 Guia de Execução Completo

Para replicar o ambiente e executar os pipelines de treinamento e previsão, siga detalhadamente os passos abaixo a partir do seu terminal.

### 1. Configuração Inicial
Clone este repositório para a sua máquina local e navegue até a pasta do projeto.
<pre><code>git clone https://github.com/SEU_USUARIO/NOME_DO_SEU_REPOSITORIO.git
cd NOME_DO_SEU_REPOSITORIO</code></pre>

### 2. Adição dos Dados
Os dados brutos não estão incluídos no repositório. **Crie a estrutura de pastas `data/raw`** e coloque os 3 arquivos de dados `.parquet` fornecidos pelo desafio dentro dela.

### 3. Ambiente e Dependências
É altamente recomendado o uso de um ambiente virtual.

<pre><code># Crie o ambiente virtual
python -m venv .venv

# Ative o ambiente (No Windows)
.venv\Scripts\activate

# Ative o ambiente (No macOS/Linux)
source .venv/bin/activate

# Instale todas as bibliotecas necessárias
pip install -r requirements.txt</code></pre>

### 4. Executando os Pipelines
Com o ambiente configurado, você pode treinar o modelo e gerar as previsões.

<ol>
  <li>
    <strong>Treinamento do Modelo:</strong><br>
    Este comando executa todo o processo: tratamento dos dados, engenharia de atributos e a otimização de hiperparâmetros com Optuna. O modelo final treinado será salvo no arquivo <code>modelo_lgbm_final.joblib</code> (ou `_xgb_`).
    <pre><code>python train.py</code></pre>
  </li>
  <li>
    <strong>Geração do Arquivo de Submissão:</strong><br>
    Este comando carrega o modelo treinado e gera o arquivo final <code>previsao_final.parquet</code>, contendo apenas as previsões para os SKUs mais relevantes para respeitar os limites da plataforma.
    <pre><code>python generate_submission.py</code></pre>
  </li>
</ol>

<br>

<details>
  <summary><strong>⚙️ Clique para ver a Metodologia Aplicada</strong></summary>

  A solução foi desenvolvida com foco em robustez e performance, incorporando técnicas avançadas de modelagem.
  <ol>
    <li>
      <strong>Processamento Robusto de Dados:</strong> Os dados foram unificados e agregados semanalmente. Uma anomalia crítica (outlier) no histórico de vendas foi identificada e corrigida para não enviesar o modelo.
    </li>
    <li>
      <strong>Engenharia de Atributos Avançada:</strong> Um conjunto rico de features foi criado para dar ao modelo um contexto profundo sobre os padrões de venda, incluindo:
        <ul>
            <li><strong>Features de Calendário Granulares:</strong> Dia do mês, dia da semana, semana do ano, etc.</li>
            <li><strong>Features de Lag:</strong> Vendas de semanas anteriores para capturar a auto-correlação.</li>
            <li><strong>Features Estatísticas de Janela Móvel:</strong> Média, desvio padrão, mínimo e máximo das vendas recentes para capturar a tendência e a volatilidade.</li>
            <li><strong>Features de Negócio:</strong> Preço médio da semana anterior e um indicador de feriados.</li>
        </ul>
    </li>
    <li>
      <strong>Transformação de Alvo (Log Transform):</strong> A variável alvo (`quantidade`) foi transformada usando <code>log(1+x)</code>. Esta técnica normaliza a distribuição de dados de vendas, permitindo que o modelo aprenda de forma mais eficaz e melhore significativamente a precisão em métricas percentuais como o WMAPE.
    </li>
    <li>
      <strong>Otimização de Hiperparâmetros:</strong> A biblioteca <strong>Optuna</strong> foi utilizada para automatizar a busca pelos melhores hiperparâmetros do modelo, executando dezenas de testes para encontrar a configuração de maior performance.
    </li>
    <li>
        <strong>Priorização Estratégica:</strong> Para atender aos limites de submissão da plataforma, foi implementada uma estratégia de priorização que foca a previsão nos 300.000 pares PDV/SKU mais relevantes com base no seu **volume de vendas recente (último trimestre de 2022)**, garantindo o maior impacto de negócio possível.
    </li>
  </ol>
</details>

