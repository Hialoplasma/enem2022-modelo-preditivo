import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline

# --- Carregar dados ---
train_df = pd.read_csv('../data/enem_2022_treino_processado.csv', sep=';', encoding='latin-1')
test_df = pd.read_csv('../data/enem_2022_teste_processado.csv', sep=';', encoding='latin-1')

# --- Criar faixas normalizadas (3 classes) ---
def criar_faixa_normalizada(nota_normalizada, limites=[0.0334, 0.3317, 0.4610]):
    #Retirando outliers abaixo das notas mínimas
    if nota_normalizada < limites[0]: 
        return -1  
    elif limites[0] <= nota_normalizada < limites[1]: 
        return 1
    elif limites[1] <= nota_normalizada < limites[2]: 
        return 2
    elif nota_normalizada >= limites[2]:
        return 3
    else:
        return -1

train_df['faixa_media'] = train_df['MEDIA_NOTAS'].apply(criar_faixa_normalizada)
test_df['faixa_media'] = test_df['MEDIA_NOTAS'].apply(criar_faixa_normalizada)

# Remover registros fora da faixa (-1)
train_df = train_df[train_df['faixa_media'] != -1].reset_index(drop=True)
test_df = test_df[test_df['faixa_media'] != -1].reset_index(drop=True)

# --- Preparar atributos e target ---
target_col = 'faixa_media'
atributo_cols = train_df.columns.difference(['MEDIA_NOTAS', target_col]).tolist()  # Convertendo para lista

X_train = train_df[atributo_cols].values.astype(np.float32)
X_test = test_df[atributo_cols].values.astype(np.float32)

# Ajustar as classes para zero-based: 1->0, 2->1, 3->2
y_train = train_df[target_col].values - 1
y_test_true = test_df[target_col].values - 1

# --- Dados originais ---
print(100 * "=")
print("Distribuição original das classes:", Counter(y_train))
print("atributos utilizadas:", atributo_cols)  # Mostrando os nomes das atributos
print(100 * "=")

# --- Balanceamento com SMOTE e undersampling ---
amostras = {0: 150000, 1: 150000, 2: 150000}

pipeline = Pipeline([
    ('oversample', SMOTE(sampling_strategy={0: 150000, 1: 150000}, random_state=42)),
    ('undersample', RandomUnderSampler(sampling_strategy={2: 150000}, random_state=42))
])

X_res, y_res = pipeline.fit_resample(X_train, y_train)

print(100 * "=")
print("Distribuição após balanceamento final:", Counter(y_res))
print(100 * "=")

# --- Configuração do XGBoost ---
class_counts = Counter(y_res)
total_samples = sum(class_counts.values())
scale_pos_weight = [total_samples / class_counts[i] for i in range(3)]
sample_weights = np.array([scale_pos_weight[y] for y in y_res])
sample_weights[y_res == 0] *= 1.3
sample_weights[y_res == 1] *= 1.5
sample_weights[y_res == 2] *= 1.0

params = {
    'objective': 'multi:softmax', #Modelo de classificação multi-classe
    'num_class': 3, # Número de classes
    'learning_rate': 0.1, # Taxa de aprendizado
    'max_depth': 8, # Profundidade máxima da árvore
    'min_child_weight': 1, # Peso mínimo da criança
    'gamma': 2, # Redução mínima da perda de função
    'subsample': 0.8, # Proporção de amostras a serem usadas para treinar cada árvore
    'colsample_bytree': 0.8, # Proporção de colunas a serem usadas para treinar cada árvore
    'reg_alpha': 0.1, # Termo de regularização L1
    'reg_lambda': 1, # Termo de regularização L2
    'n_estimators': 100, # Número de árvores
    'random_state': 42, # Semente para reprodutibilidade
    'scale_pos_weight': scale_pos_weight, # Pesos para classes desbalanceadas
    'tree_method': 'hist', # Método de construção da árvore
    'enable_categorical': False # Habilitar suporte a variáveis categóricas
}

# Criar e treinar o modelo
model = XGBClassifier(**params)
model.fit(
    X_res, 
    y_res,
    sample_weight=sample_weights
)

# --- Avaliação ---
y_pred = model.predict(X_test)

print("\nClassification report (XGBoost Model):")
print(classification_report(y_test_true, y_pred, digits=4))

# Matriz de confusão
class_names = ['350 a 510', '510 a 600', '600 +']
cm = confusion_matrix(y_test_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão - XGBoost')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.savefig('../reports/images/matriz_confusao_xgboost.png')
plt.show()
plt.close()

# Salvar o modelo
model.save_model('../models/modelo_xgboost.json')

# Obter a importância das atributos
importancia = model.get_booster().get_score(importance_type='weight')

# Criar um DataFrame com os nomes reais das atributos
importancia_df = pd.DataFrame({
    'atributo': [atributo_cols[int(f[1:])] for f in importancia.keys()],  # Converte f1, f2 para nomes reais
    'importancia': list(importancia.values())
}).sort_values('importancia', ascending=False)

# Plotar o gráfico
plt.figure(figsize=(12, 8))
sns.barplot(x='importancia', y='atributo', data=importancia_df.head(20), palette='Blues_r')
plt.title('atributo importancia - XGBoost (Top 20 atributos)')
plt.xlabel('importancia Score')
plt.ylabel('atributos')
plt.tight_layout()
plt.savefig('../reports/images/atributo_importancia.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nTop 7 atributos:")
print(importancia_df.head(7).to_string(index=False))

# atributo importancia em formato de tabela
importancia = model.get_booster().get_score(importance_type='weight')
importancia_df = pd.DataFrame({
    'atributo': [atributo_cols[int(f[1:])] for f in importancia.keys()],  # Converte f1, f2, etc para nomes reais
    'importancia': list(importancia.values())
}).sort_values('importancia', ascending=False)
