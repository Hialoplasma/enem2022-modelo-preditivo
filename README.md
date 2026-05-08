# Predição de Desempenho no ENEM via Variáveis Socioeconômicas

Este projeto apresenta um modelo de inteligência artificial desenvolvido para classificar o desempenho acadêmico de estudantes concluintes do Ensino Médio no ENEM 2022, utilizando exclusivamente indicadores socioeconômicos. O trabalho foi realizado como requisito para a disciplina de Inteligência Artificial na Universidade Federal de São João del-Rei (UFSJ).

## 1. Descrição do Problema
O objetivo central é investigar em que medida fatores externos ao ambiente escolar — como renda familiar, raça e tipo de escola — correlacionam-se com a nota final dos candidatos. O modelo atua como um classificador que segmenta os estudantes em três faixas de desempenho: 
* **Baixo desempenho**: 350-510 pontos.
* **Médio desempenho**: 510-600 pontos.
* **Alto desempenho**: acima de 600 pontos.

## 2. Estrutura do Repositório
* **src/**: Contém o código-fonte do projeto.
    * `pre_processamento_Enem_2022.py`: Rotina de limpeza, filtragem de candidatos e normalização dos microdados brutos.
    * `XGBoost.py`: Script de treinamento utilizando o algoritmo XGBoost, incluindo técnicas de balanceamento de classes (SMOTE/Undersampling) e geração de métricas.
    * `consulta.py`: Interface de linha de comando para predição individual baseada em perfis de estudantes.
* **data/**: Diretório reservado para os dados. Contém instruções para download dos microdados originais (INEP) e os arquivos CSV processados.
* **models/**: Armazena os artefatos gerados pelo treinamento.
    * `modelo_xgboost.json`: O modelo persistido para uso em consultas.
    * `renda_scaler.pkl` e `notas_scaler.pkl`: Objetos para normalização de valores numéricos.
* **reports/**: Documentação técnica e visualizações.
    * `Relatorio_Tecnico_ENEM.pdf`: Documento acadêmico detalhando a metodologia e fundamentação teórica.
    * `figuras/`: Matrizes de confusão, correlação e gráficos de importância de atributos.

## 3. Metodologia e Tecnologias
O pipeline foi construído utilizando as seguintes tecnologias e técnicas:
* **Linguagem**: Python 3.11+.
* **Processamento**: Pandas e NumPy para manipulação de grandes volumes de dados.
* **Algoritmo**: XGBoost Classifier com objetivo `multi:softmax`.
* **Tratamento de Dados**: Scikit-learn (StandardScaler e MinMaxScaler) e Imbalanced-learn (SMOTE/Pipeline).
* **Visualização**: Matplotlib e Seaborn para geração de gráficos analíticos.

## 4. Instalação e Execução

### Pré-requisitos
Instale as dependências listadas no arquivo `requirements.txt`:
    pip install -r requirements.txt

Fluxo de ExecuçãoDados: 

1 - Obtenha os microdados de 2022 no site do INEP e posicione o arquivo na pasta data/.
    
2 - Processamento: Execute o script de pré-processamento para gerar as bases de treino e    teste processadas:
    python src/pre_processamento_Enem_2022.py

3 - Treinamento: Execute o treinamento para gerar o modelo e as métricas de validação:
    python src/XGBoost.py

4 - Consulta: Utilize o script de consulta para testar o modelo com perfis customizados:
    python src/consulta.py

5 -  AutoriaTrabalho desenvolvido por estudantes do Departamento de Ciência da Computação - UFSJ. Detalhes sobre a fundamentação sociológica e os resultados estatísticos podem ser consultados no relatório técnico na pasta reports/.  