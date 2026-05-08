import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Iniciando pré-processamento dos dados do ENEM 2022...")
# Configurações iniciais
pd.set_option('display.max_columns', 50)
plt.style.use('ggplot')

# Carregar dados
df = pd.read_csv('../data/MICRODADOS_ENEM_2022.csv', sep=';', encoding='latin-1', low_memory=True)

# Pré-processamento inicial
def preprocess_data(df):
    # Tratar valores ausentes
    df.fillna(0, inplace=True)
    
    # Filtrar dados
    df = df[
        (df['TP_COR_RACA'] != 0) &
        (df['TP_PRESENCA_CN'] == 1) &
        (df['TP_PRESENCA_CH'] == 1) &
        (df['TP_PRESENCA_LC'] == 1) &
        (df['TP_PRESENCA_MT'] == 1) &
        (df['IN_TREINEIRO'] == 0) &
        (df['TP_ST_CONCLUSAO'] == 2) &
        (df['TP_DEPENDENCIA_ADM_ESC'] != 0) &   
        (df['TP_ENSINO'] == 1) & #Apenas ensino regular devido a baixa amostra de outros tipos
        ((df['TP_FAIXA_ETARIA'] > 1) & (df['TP_FAIXA_ETARIA'] <= 4)) & # 17 a 19 anos
        (df['TP_COR_RACA'] < 4)& #Brancos, pardos e pretos
        (df['TP_LOCALIZACAO_ESC'] == 1) #Apenas escolas urbanas
    ]

    # Calcular média das notas
    colunas_notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']

    # 1. Criar máscara booleana para filtrar linhas onde TODAS as notas > 300
    mascara_notas_validas = (df[colunas_notas] > 300).all(axis=1)

    # 2. Aplicar o filtro ao DataFrame (remover linhas que não atendem à condição)
    df = df[mascara_notas_validas].copy()

    # 3. Calcular a média apenas para as linhas filtradas
    df['MEDIA_NOTAS'] = df[colunas_notas].mean(axis=1)
    print("Menor nota média:", df['MEDIA_NOTAS'].min())
    print("Maior nota média:", df['MEDIA_NOTAS'].max())

    # Remover colunas desnecessárias
    colunas_remover = ['TP_LOCALIZACAO_ESC','TP_ESCOLA',   'NU_INSCRICAO', 'TP_ESTADO_CIVIL', 'NU_ANO', 'TP_NACIONALIDADE', 
                       'IN_TREINEIRO', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TX_GABARITO_CN', 
                       'TX_GABARITO_CH', 'TX_GABARITO_LC', 'TX_GABARITO_MT', 'TX_RESPOSTAS_LC', 
                       'TX_RESPOSTAS_MT', 'TX_RESPOSTAS_CH', 'TP_LINGUA', 'TP_STATUS_REDACAO', 
                       'TX_RESPOSTAS_CN', 'CO_MUNICIPIO_ESC', 'NO_MUNICIPIO_ESC', 'SG_UF_PROVA', 
                       'SG_UF_ESC', 'TP_SIT_FUNC_ESC', 'CO_MUNICIPIO_PROVA', 'NO_MUNICIPIO_PROVA', 
                       'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT', 'NU_NOTA_COMP1', 
                       'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 
                       'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 
                       'NU_NOTA_REDACAO', 'NU_NOTA_CN', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_CH', 
                       'CO_UF_PROVA', 'TP_ENSINO', 'CO_UF_ESC'] + [col for col in df.columns if col.startswith('Q') and col != 'Q006']

    df.drop(columns=colunas_remover, inplace=True)
    
    # Mapeamento de variáveis
    mapeamento_renda = {
        'A': 0, 'B': (0+1212)/2, 'C': (1212+1818)/2, 'D': (1818+2424)/2,
        'E': (2424+3030)/2, 'F': (3030+3636)/2, 'G': (3636+4848)/2,
        'H': (4848+6060)/2, 'I': (6060+7272)/2, 'J': (7272+8484)/2,
        'K': (8484+9696)/2, 'L': (9696+10908)/2, 'M': (10908+12120)/2,
        'N': (12120+14544)/2, 'O': (14544+18180)/2, 'P': (18180+24240)/2,
        'Q': 24240*1.5
    }

    df['RENDA_FAMILIAR'] = np.log1p(df['Q006'].map(mapeamento_renda))
    df.drop(columns=['Q006'], inplace=True) #Drop de todas as colunas do questionário que não são relevantes para a análise

    df['TP_SEXO'] = df['TP_SEXO'].map({'F': 1, 'M': 2})   # 1=Feminino, 2=Masculino     
    
    # Selecionar e ordenar colunas
    cols = ['TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_COR_RACA', 'TP_DEPENDENCIA_ADM_ESC',
            'RENDA_FAMILIAR', 'MEDIA_NOTAS']
    
    return df[cols]

df = preprocess_data(df)

# Análise exploratória inicial
print("Distribuição inicial das variáveis:")
print(df.describe())

df.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.savefig('../reports/images/distribuicoes_iniciais.png')

# Divisão treino-teste ANTES de qualquer transformação
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Cálculo da mediana federal (BRUTA) UMA VEZ com os dados de treino ---
# Este valor será salvo e usado na consulta
mediana_federal_bruta = train_df.loc[train_df['TP_DEPENDENCIA_ADM_ESC'] == 1, 'MEDIA_NOTAS'].median()
print(f"Mediana das notas das escolas federais (bruta): {mediana_federal_bruta:.2f}")


# Função para processamento final após separação dos dados entre treino e teste
def processamento_final(df_segment, scaler_renda_obj=None, scaler_notas_obj=None, fit_scalers=False, median_federal=None):
    # One-hot encoding para variáveis categóricas
    categorical_cols = ['TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_COR_RACA']
    df_segment = pd.get_dummies(df_segment, columns=categorical_cols, drop_first=True)

    # Tratamento especial para TP_DEPENDENCIA_ADM_ESC assumindo hierarquia de desempenho
    ordem = {3: 1, 2: 2, 1: 3, 4: 4}   # Municipal(3) -> Privada(4)
    
    df_segment['DEPENDENCIA_ORD'] = df_segment['TP_DEPENDENCIA_ADM_ESC'].map(ordem)
    df_segment.drop(columns=['TP_DEPENDENCIA_ADM_ESC'], inplace=True)
    # Tratamento especial para RENDA_FAMILIAR
    if fit_scalers:
        scaler_renda_obj = StandardScaler()
        scaler_notas_obj = MinMaxScaler()
        
        # Ajustar scalers apenas com dados de treino
        df_segment['RENDA_FAMILIAR'] = scaler_renda_obj.fit_transform(df_segment[['RENDA_FAMILIAR']])
        df_segment['MEDIA_NOTAS'] = scaler_notas_obj.fit_transform(df_segment[['MEDIA_NOTAS']])
    else:
        # Aplicar transformação com scalers pré-ajustados
        df_segment['RENDA_FAMILIAR'] = scaler_renda_obj.transform(df_segment[['RENDA_FAMILIAR']])
        df_segment['MEDIA_NOTAS'] = scaler_notas_obj.transform(df_segment[['MEDIA_NOTAS']])
    
    return df_segment, scaler_renda_obj, scaler_notas_obj

# Processar dados de treino (com ajuste de scalers e passando a mediana bruta)
train_df, renda_scaler, notas_scaler = processamento_final(train_df, fit_scalers=True, median_federal=mediana_federal_bruta)

# Processar dados de teste (com scalers já ajustados e passando a mediana bruta)
test_df, _, _ = processamento_final(test_df, renda_scaler, notas_scaler, median_federal=mediana_federal_bruta)

# --- SALVAR OS OBJETOS NECESSÁRIOS PARA A CONSULTA ---
joblib.dump(renda_scaler, '../models/renda_scaler.pkl')
joblib.dump(notas_scaler, '../models/notas_scaler.pkl')
joblib.dump(mediana_federal_bruta, '../models/mediana_federal_bruta.pkl')
print("\nArquivos 'renda_scaler.pkl', 'notas_scaler.pkl' e 'mediana_federal_bruta.pkl' salvos com sucesso!")

# Verificação final
print("\nDimensões dos datasets:")
print(f"Treino: {train_df.shape}, Teste: {test_df.shape}")

print("\nPrimeiras linhas do dataset de treino:")
print(train_df.head())

# Visualização das distribuições após transformação
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['RENDA_FAMILIAR'], kde=True)
plt.title('Distribuição de Renda (Padronizada)')

plt.subplot(1, 2, 2)
sns.histplot(train_df['MEDIA_NOTAS'], kde=True)
plt.title('Distribuição de Notas (Normalizada)')
plt.tight_layout()
plt.savefig('../reports/images/distribuicoes_numericas.png')

# Matriz de correlação
plt.figure(figsize=(15, 10))
sns.heatmap(train_df.corr(), cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.savefig('../reports/images/matriz_correlacao.png')

# Análise das variáveis categóricas
# As colunas categóricas originais foram transformadas em One-Hot Encoding ou ordinais.
categorical_vars_transformed = [col for col in train_df.columns if 'TP_FAIXA_ETARIA' in col or 'TP_SEXO' in col or 'TP_COR_RACA' in col]
categorical_vars_transformed.append('DEPENDENCIA_ORD')


plt.figure(figsize=(15, 20))
# Filtra apenas as colunas que existem no DataFrame após o processamento
existent_transformed_cols = [col for col in categorical_vars_transformed if col in train_df.columns]
for i, var in enumerate(existent_transformed_cols, 1):
    plt.subplot(len(existent_transformed_cols)//2 + 1, 2, i) # Ajusta o layout dinamicamente
    train_df[var].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'Distribuição de {var}')
    plt.ylabel('Proporção')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../reports/images/distribuicoes_categoricas_transformadas.png')
plt.close()


# Análise das variáveis numéricas
numeric_vars = ['RENDA_FAMILIAR', 'MEDIA_NOTAS']

plt.figure(figsize=(12, 6))
for i, var in enumerate(numeric_vars, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(x=train_df[var]) 
    plt.title(f'Boxplot de {var}')
plt.tight_layout()
plt.savefig('../reports/images/boxplots_numericos.png')
plt.close()

# Estatísticas descritivas
print("\nEstatísticas descritivas das variáveis numéricas (após transformação):")
print(train_df[numeric_vars].describe())

print("\nContagem de categorias transformadas:")
for var in existent_transformed_cols:
    print(f"\n{var}:")
    print(train_df[var].value_counts())

# Salvar datasets processados
train_df.to_csv('../data/enem_2022_treino_processado.csv', sep=';', index=False, encoding='latin-1')
test_df.to_csv('../data/enem_2022_teste_processado.csv', sep=';', index=False, encoding='latin-1')

print("\nPré-processamento concluído.")
print(f"Total de features geradas: {train_df.shape[1]}")