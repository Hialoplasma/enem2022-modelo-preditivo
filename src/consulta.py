import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# --- Carregar o modelo treinado ---
model = xgb.XGBClassifier()
try:
    model.load_model('../models/modelo_xgboost.json')
    print("Modelo 'modelo_xgboost.json' carregado com sucesso.")
except xgb.core.XGBoostError:
    print("Erro ao carregar o modelo. Verifique se 'modelo_xgboost.json' existe e está correto.")
    exit()

# --- Carregar os scalers ---
try:
    renda_scaler = joblib.load('../models/renda_scaler.pkl')
    print("Scalers carregados com sucesso.")
except FileNotFoundError:
    print("Erro: 'renda_scaler.pkl' não encontrado.")
    print("Certifique-se de executar o script de pré-processamento para gerá-los.")
    exit()

PARAMETROS = [
    'DEPENDENCIA_ORD',
    'RENDA_FAMILIAR',
    'TP_COR_RACA_2',
    'TP_COR_RACA_3',
    'TP_FAIXA_ETARIA_3',
    'TP_FAIXA_ETARIA_4',
    'TP_SEXO_2'
]

# Classes de saída
CLASSES = ['350 a 510 (Baixo)', '510 a 600 (Médio)', '600 + (Alto)']

# Mapeamento de renda
mapeamento_renda = {
    'A': 0, 'B': (0+1212)/2, 'C': (1212+1818)/2, 'D': (1818+2424)/2,
    'E': (2424+3030)/2, 'F': (3030+3636)/2, 'G': (3636+4848)/2,
    'H': (4848+6060)/2, 'I': (6060+7272)/2, 'J': (7272+8484)/2,
    'K': (8484+9696)/2, 'L': (9696+10908)/2, 'M': (10908+12120)/2,
    'N': (12120+14544)/2, 'O': (14544+18180)/2, 'P': (18180+24240)/2,
    'Q': 24240*1.5
}

def preprocessamento_entrada(atributo, renda_scaler):
    """
    Transforma os dados de entrada brutos em parametros prontos para o modelo.
    """
    atributo_processado = {}

    # 1. DEPENDENCIA_ORD
    dependencia_map = {'Municipal': 1, 'Estadual': 2, 'Federal': 3, 'Privada': 4}
    atributo_processado['DEPENDENCIA_ORD'] = dependencia_map.get(atributo['dependencia_escola'], 0)
    if atributo_processado['DEPENDENCIA_ORD'] == 0:
        raise ValueError(f"Dependência de escola inválida: {atributo['dependencia_escola']}. Use 'Municipal', 'Estadual', 'Federal' ou 'Privada'.")

    # 2. RENDA_FAMILIAR (Q006)
    if atributo['renda_q006'] not in mapeamento_renda:
        raise ValueError(f"Categoria de renda inválida: {atributo['renda_q006']}. Use A, B, C, ..., Q.")
    renda_bruta = mapeamento_renda[atributo['renda_q006']]
    renda_log = np.log1p(renda_bruta)
    atributo_processado['RENDA_FAMILIAR'] = renda_scaler.transform([[renda_log]])[0][0]

    # 3. TP_COR_RACA_2 e TP_COR_RACA_3 (One-Hot Encoded)
    atributo_processado['TP_COR_RACA_2'] = 1 if atributo['cor_raca'].lower() == 'preta' else 0
    atributo_processado['TP_COR_RACA_3'] = 1 if atributo['cor_raca'].lower() == 'parda' else 0
    if atributo['cor_raca'].lower() not in ['branca', 'preta', 'parda']:
        raise ValueError(f"Cor/Raça inválida: {atributo['cor_raca']}. Use 'Branca', 'Preta' ou 'Parda'.")

    # 4. TP_FAIXA_ETARIA_3 e TP_FAIXA_ETARIA_4 (One-Hot Encoded)
    atributo_processado['TP_FAIXA_ETARIA_3'] = 1 if atributo['idade'] == 18 else 0
    atributo_processado['TP_FAIXA_ETARIA_4'] = 1 if atributo['idade'] == 19 else 0
    if atributo['idade'] not in [17, 18, 19]:
        raise ValueError(f"Idade inválida: {atributo['idade']}. Use 17, 18 ou 19.")

    # 5. TP_SEXO_2 (One-Hot Encoded)
    atributo_processado['TP_SEXO_2'] = 1 if atributo['sexo'].lower() == 'masculino' else 0
    if atributo['sexo'].lower() not in ['masculino', 'feminino']:
        raise ValueError(f"Sexo inválido: {atributo['sexo']}. Use 'Masculino' ou 'Feminino'.")

    return atributo_processado

def calc_faixa_notas(entrada):
    """
    Calcula as probabilidades de classificação do estudante com base nas características fornecidas.    
    Args:
        entrada (dict): Dicionário com os valores para cada característica do estudante:
            - 'idade' (int): Idade do estudante (17, 18 ou 19)
            - 'sexo' (str): 'Feminino' ou 'Masculino'
            - 'cor_raca' (str): 'Branca', 'Preta' ou 'Parda'
            - 'dependencia_escola' (str): 'Federal', 'Estadual', 'Municipal' ou 'Privada'
            - 'renda_q006' (str): Categoria de renda do Q006 (A, B, C, ..., Q)
            
    Retorna:
        DataFrame: Probabilidades para cada classe
    """
    # Pré-processar os dados de entrada
    preprocessamentoEntrada = preprocessamento_entrada(entrada, renda_scaler)
    
    # Criar DataFrame com os dados pré-processados
    df = pd.DataFrame([preprocessamentoEntrada])

    # Garantir que todas as parâmetros estão presentes e na ordem correta
    df = df[PARAMETROS]

    # Fazer a predição
    probabilidades = model.predict_proba(df)[0]
    
    # Criar DataFrame com os resultados
    result = pd.DataFrame({
        'Classe': CLASSES,
        'Probabilidade (%)': probabilidades * 100
    })
    
    return result

def plot_probabilidades(probabilidades_df, entrada):
    """
    Gera um gráfico de barras com as probabilidades, incluindo os parâmetros de entrada.
    
    Args:
        probabilidades_df (DataFrame): DataFrame com as probabilidades
        entrada (dict): Dicionário com os valores de entrada intuitivos.
    """
    # Mapeamento completo das faixas de renda para exibição
    mapeamento_renda_descricao = {
        'A': "Nenhuma renda",
        'B': "Até R$ 1.212",
        'C': "De R$ 1.212,01 até R$ 1.818",
        'D': "De R$ 1.818,01 até R$ 2.424",
        'E': "De R$ 2.424,01 até R$ 3.030",
        'F': "De R$ 3.030,01 até R$ 3.636",
        'G': "De R$ 3.636,01 até R$ 4.848",
        'H': "De R$ 4.848,01 até R$ 6.060",
        'I': "De R$ 6.060,01 até R$ 7.272",
        'J': "De R$ 7.272,01 até R$ 8.484",
        'K': "De R$ 8.484,01 até R$ 9.696",
        'L': "De R$ 9.696,01 até R$ 10.908",
        'M': "De R$ 10.908,01 até R$ 12.120",
        'N': "De R$ 12.120,01 até R$ 14.544",
        'O': "De R$ 14.544,01 até R$ 18.180",
        'P': "De R$ 18.180,01 até R$ 24.240",
        'Q': "Acima de R$ 24.240"
    }
    
    # Obter a descrição da renda
    renda_descricao = mapeamento_renda_descricao.get(entrada['renda_q006'], "Renda desconhecida")
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(probabilidades_df['Classe'], probabilidades_df['Probabilidade (%)'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Formatar o título do gráfico com os parâmetros de entrada
    title_text = f"Probabilidade de Classificação do Estudante\n" \
                 f"Idade: {entrada['idade']} | " \
                 f"Sexo: {entrada['sexo']} | " \
                 f"Cor/Raça: {entrada['cor_raca']}\n" \
                 f"Dependência da Escola: {entrada['dependencia_escola']} | " \
                 f"Renda: {renda_descricao}"
                 
    plt.title(title_text, fontsize=14)
    plt.xlabel('Faixa de Nota', fontsize=12)
    plt.ylabel('Probabilidade (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Adicionar os valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


"""        
'A': "Nenhuma renda",
'B': "Até R$ 1.212",
'C': "De R$ 1.212,01 até R$ 1.818",
'D': "De R$ 1.818,01 até R$ 2.424",
'E': "De R$ 2.424,01 até R$ 3.030",
'F': "De R$ 3.030,01 até R$ 3.636",
'G': "De R$ 3.636,01 até R$ 4.848",
'H': "De R$ 4.848,01 até R$ 6.060",
'I': "De R$ 6.060,01 até R$ 7.272",
'J': "De R$ 7.272,01 até R$ 8.484",
'K': "De R$ 8.484,01 até R$ 9.696",
'L': "De R$ 9.696,01 até R$ 10.908",
'M': "De R$ 10.908,01 até R$ 12.120",
'N': "De R$ 12.120,01 até R$ 14.544",
'O': "De R$ 14.544,01 até R$ 18.180",
'P': "De R$ 18.180,01 até R$ 24.240",
'Q': "Acima de R$ 24.240"""


# --------------------------------------------- CONSULTA -------------------------------------------------
if __name__ == "__main__":
    entrada = {
        'idade': 18,
        'sexo': 'Masculino',
        'cor_raca': 'Preta',
        'dependencia_escola': 'Privada',
        'renda_q006': 'I', 
    }

    print("Entrada do Estudante:")
    idade = int(input("Idade (17, 18 ou 19): "))
    sexo = input("Sexo (Masculino ou Feminino): ")
    cor_raca = input("Cor/Raça (Branca, Preta ou Parda): ")
    dependencia_escola = input("Dependência da Escola (Federal, Estadual, Municipal ou Privada): ")
    renda_q006 = input("Renda Familiar (A, B, C, ..., Q): ")
    entrada = {
        'idade': idade,
        'sexo': sexo,
        'cor_raca': cor_raca,
        'dependencia_escola': dependencia_escola,
        'renda_q006': renda_q006
    }
    
    try:
        resultados = calc_faixa_notas(entrada)
        
        print("\nProbabilidades de Classificação:")
        print(resultados.to_string(index=False))
        
        plot_probabilidades(resultados, entrada)
    except ValueError as e:
        print(f"Erro na entrada: {e}")