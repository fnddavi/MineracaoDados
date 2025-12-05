import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CARREGAR E PREPARAR OS DADOS
# ============================================================================

# Carregar dataset
# Certifique-se de que o arquivo 'breast-cancer.csv' está na mesma pasta
try:
    df = pd.read_csv('breast-cancer.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'breast-cancer.csv' não encontrado. Verifique se ele está na pasta atual.")
    exit()

# Substituir '?' por NaN
df_clean = df.copy()
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].replace('?', np.nan)

print("="*80)
print("🤖 QUESTÃO 6 - CLASSIFICAÇÃO (Seguindo EXATAMENTE as instruções)")
print("="*80)

# Separar features e target
X = df_clean.drop('Class', axis=1)
y = df_clean['Class']

# Codificar a variável target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"📊 Dimensões:")
print(f"   • X (features): {X.shape}")
print(f"   • y (target): {y.shape}")
print(f"   • Classes: {np.unique(y, return_counts=True)}")

# ============================================================================
# MÉTODO 1: EXATAMENTE como a professora exemplificou
# ============================================================================

print("\n" + "="*80)
print("📝 MÉTODO 1: Aplicando OneHotEncoder EXATAMENTE como no exemplo da professora")
print("="*80)

# Identificar colunas categóricas (todas exceto 'deg-malig' que é numérica)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"📋 Colunas categóricas (que serão convertidas para binário):")
for col in categorical_cols:
    unique_vals = X[col].dropna().unique()[:5]  # Mostrar primeiros 5 valores únicos
    print(f"   • {col}: {len(X[col].dropna().unique())} valores únicos, ex: {list(unique_vals)}")

# PASSO 1: Substituir valores ausentes pela MODA
print("\n🔧 PASSO 1: Substituindo valores ausentes pela MODA...")
for col in X.columns:
    if X[col].isnull().any():
        moda = X[col].mode()[0] if not X[col].mode().empty else 'unknown'
        X[col] = X[col].fillna(moda)
        print(f"   • {col}: substituídos {X[col].isnull().sum()} valores pela moda '{moda}'")

# Verificar se ainda há valores ausentes
print(f"\n✅ Após imputação pela moda: {X.isnull().sum().sum()} valores ausentes restantes")

# PASSO 2: Aplicar OneHotEncoder EXATAMENTE como no exemplo da professora
print("\n🔧 PASSO 2: Aplicando OneHotEncoder...")
print("   encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)")

# Criar o encoder exatamente como no exemplo
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

# Aplicar transformação (apenas nas colunas categóricas)
# Primeiro, vamos codificar apenas as colunas categóricas
X_categorical = X[categorical_cols]

# Fit e transform
print("   X_encoded = encoder.fit_transform(X_categorical).toarray()")
X_encoded = encoder.fit_transform(X_categorical).toarray()

print(f"\n📊 Resultado da codificação:")
print(f"   • Forma original de X_categorical: {X_categorical.shape}")
print(f"   • Forma após OneHotEncoder: {X_encoded.shape}")
print(f"   • Número de features binárias criadas: {X_encoded.shape[1]}")

# Agora, precisamos combinar com as colunas numéricas (deg-malig)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n🔢 Colunas numéricas (não convertidas para binário): {numeric_cols}")

if numeric_cols:
    X_numeric = X[numeric_cols].values
    # Concatenar features numéricas com features codificadas
    X_final = np.hstack([X_numeric, X_encoded])
    print(f"   • Forma final de X (numéricas + categóricas codificadas): {X_final.shape}")
else:
    X_final = X_encoded

# ============================================================================
# CLASSIFICAÇÃO COM CROSS-VALIDATION (10 FOLDS)
# ============================================================================

print("\n" + "="*80)
print("🎯 CLASSIFICAÇÃO COM CROSS-VALIDATION (10 FOLDS)")
print("="*80)

# Definir os classificadores
j48 = DecisionTreeClassifier(random_state=42, criterion='entropy')
naive_bayes = GaussianNB()

# Configurar cross-validation (10 folds)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"\n🔧 Configuração da validação cruzada:")
print(f"   • Número de folds: 10")
print(f"   • Total de instâncias: {len(X_final)}")
print(f"   • Total de features: {X_final.shape[1]}")

# Avaliar J48
print("\n📊 Avaliando J48 (Árvore de Decisão)...")
j48_scores = cross_val_score(j48, X_final, y_encoded, cv=cv, scoring='accuracy')
j48_mean = j48_scores.mean()
j48_std = j48_scores.std()

print(f"   • Scores por fold: {j48_scores}")
print(f"   • Acurácia média: {j48_mean:.4f} ({j48_mean*100:.2f}%)")
print(f"   • Desvio padrão: {j48_std:.4f}")

# Avaliar Naive Bayes
print("\n📊 Avaliando Naive Bayes...")
nb_scores = cross_val_score(naive_bayes, X_final, y_encoded, cv=cv, scoring='accuracy')
nb_mean = nb_scores.mean()
nb_std = nb_scores.std()

print(f"   • Scores por fold: {nb_scores}")
print(f"   • Acurácia média: {nb_mean:.4f} ({nb_mean*100:.2f}%)")
print(f"   • Desvio padrão: {nb_std:.4f}")

# ============================================================================
# RESULTADO FINAL
# ============================================================================

print("\n" + "="*80)
print("🏆 RESULTADO FINAL - QUESTÃO 6")
print("="*80)

print(f"\n📈 DESEMPENHO DOS ALGORITMOS:")
print("-" * 50)

# Criar DataFrame para visualização
results_df = pd.DataFrame({
    'Algoritmo': ['J48 (Árvore de Decisão)', 'Naive Bayes'],
    'Acurácia Média': [j48_mean, nb_mean],
    'Desvio Padrão': [j48_std, nb_std],
    'Acurácia (%)': [f"{j48_mean*100:.2f}%", f"{nb_mean*100:.2f}%"]
})

print(results_df.to_string(index=False))

# Determinar o melhor algoritmo
print("\n" + "="*50)
print("🎖️  MELHOR ALGORITMO:")
print("="*50)

if j48_mean > nb_mean:
    print(f"✅ J48 (Árvore de Decisão) apresentou o MELHOR resultado!")
    print(f"   • Acurácia: {j48_mean*100:.2f}%")
    print(f"   • Superioridade: {abs(j48_mean - nb_mean)*100:.2f}% sobre Naive Bayes")
elif nb_mean > j48_mean:
    print(f"✅ Naive Bayes apresentou o MELHOR resultado!")
    print(f"   • Acurácia: {nb_mean*100:.2f}%")
    print(f"   • Superioridade: {abs(nb_mean - j48_mean)*100:.2f}% sobre J48")
else:
    print("⚠️  Os dois algoritmos apresentaram resultados equivalentes!")

# Visualização comparativa (Salvar em arquivo em vez de mostrar na tela)
plt.figure(figsize=(12, 6))

# Gráfico 1: Acurácia média
plt.subplot(1, 2, 1)
algorithms = ['J48', 'Naive Bayes']
accuracies = [j48_mean, nb_mean]
colors = ['skyblue', 'lightcoral']

bars = plt.bar(algorithms, accuracies, color=colors, alpha=0.8, edgecolor='black')
plt.title('Acurácia Média dos Algoritmos', fontsize=14, fontweight='bold')
plt.ylabel('Acurácia', fontsize=12)
plt.ylim([0, 1.0])
plt.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}\n({acc*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

# Gráfico 2: Scores por fold
plt.subplot(1, 2, 2)
folds = range(1, 11)
plt.plot(folds, j48_scores, 'o-', color='skyblue', linewidth=2, markersize=8, label='J48')
plt.plot(folds, nb_scores, 's-', color='lightcoral', linewidth=2, markersize=8, label='Naive Bayes')
plt.title('Scores por Fold (10-Fold CV)', fontsize=14, fontweight='bold')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.xticks(folds)
plt.ylim([0, 1.0])
plt.grid(True, alpha=0.3)
plt.legend()

plt.suptitle('Comparação de Algoritmos de Classificação', fontsize=16, fontweight='bold')
plt.tight_layout()

# Salvar o gráfico
plt.savefig('resultado_comparacao.png')
print("\n📊 Gráfico salvo como 'resultado_comparacao.png'")
