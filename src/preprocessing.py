import pandas as pd
import numpy as np
import scipy 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pyampute.exploration.mcar_statistical_tests import MCARTest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# Carga del dataset
data = pd.read_csv("water_potability.csv")

# Visualización de las primeras observaciones
print(data.head())

# Verificación el tamaño del dataset
print(data.shape)

# Obtención de información relevante
print(data.info)

# Búsqueda de datos faltantes
cantidad_nas = data.isna().sum().sort_values(ascending=False)

print(f"{'-'*5} VARIABLES ORDENADAS POR LAS QUE MÁS NA'S CONTIENEN {'-'*5}")
print(cantidad_nas)

porcentaje_nas = ((data.isna().sum()/len(data))*100).sort_values(ascending=False)

print(f"{'-'*5} VARIABLES ORDENADAS POR LAS QUE MÁS NA'S CONTIENEN (%) {'-'*5}")
print(porcentaje_nas)

# Test de Little
test = MCARTest(method="little")
p_valor = test.little_mcar_test(data)

print(f"El p-valor obtenido es: {p_valor:.2f}.")

# Imputación de valores faltantes
# Definimos las columnas que contienen NA's
columnas_nas = ["Sulfate", "ph", "Trihalomethanes"]
# Definimos que la imputacion se va a realizar en base a la mediana
imputacion = SimpleImputer(strategy="median")

# Imputación
data[columnas_nas] = imputacion.fit_transform(data[columnas_nas])

# Datos tras la imputación
print(data.head())

# Verificación de la inexistencia de datos faltantes
data.isna().sum()


# Tratamiento de outliers
dicc_outliers = {}

for columna in data.columns:
    z = np.abs(scipy.stats.zscore(data[columna]))
    dicc_outliers[columna] = (z > 3).sum()

# Mostramos los outliers de forma ordenada:
outliers_ordenados = pd.Series(dicc_outliers).sort_values(ascending=False)

print(f"{'-'*5} CANTIDAD DE OUTLIERS POR COLUMNA (USANDO Z-SCORE) {'-'*5}")
print(outliers_ordenados)


# Análisis de la cantidad de outliers usando diagramas de cajas
columnas_con_outliers = ['ph', 'Hardness', 'Solids', 'Chloramines', 
                          'Sulfate', 'Conductivity', 'Organic_carbon', 
                          'Trihalomethanes', 'Turbidity']

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(columnas_con_outliers):
    axes[i].boxplot(data[col].dropna())
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# Análisis de la existencia de outliers usando Z-Score
rangos_validos = {
    'ph':               (0, 14),
    'Hardness':         (0, 500),
    'Solids':           (0, 50000),
    'Chloramines':      (0, 10),
    'Sulfate':          (0, 500),
    'Conductivity':     (0, 1000),
    'Organic_carbon':   (0, 30),
    'Trihalomethanes':  (0, 120),
    'Turbidity':        (0, 10)
}

columnas_winsorizacion = []
columnas_mantener = []
columnas_eliminar = []

for col in columnas_con_outliers:
    z = np.abs(scipy.stats.zscore(data[col].dropna()))
    mascara = z > 3

    num_outliers = mascara.sum()
    porcentaje = (num_outliers / len(data)) * 100
    valores_outlier = data[col].dropna()[mascara]
    min_val = valores_outlier.min()
    max_val = valores_outlier.max()
    media_outlier = data.loc[mascara, 'Potability'].mean()
    media_normal = data.loc[~mascara, 'Potability'].mean()
    diferencia = abs(media_outlier - media_normal)

    rango_min, rango_max = rangos_validos[col]
    fuera_de_rango = valores_outlier[(valores_outlier < rango_min) | (valores_outlier > rango_max)]

    print(f"{'='*50}")
    print(f"Variable: '{col}'")
    print(f"  Nº outliers:         {num_outliers} ({porcentaje:.2f}%)")
    print(f"  Rango outliers:      [{min_val:.2f}, {max_val:.2f}]")
    print(f"  Potability outliers: {media_outlier:.2f}")
    print(f"  Potability normales: {media_normal:.2f}")
    print(f"  Diferencia:          {diferencia:.2f}")

    if len(fuera_de_rango) > 0:
        print(f"  {len(fuera_de_rango)} outliers FUERA del rango válido ({rango_min}, {rango_max})")
        print("  Decisión: ELIMINAR")
        columnas_eliminar.append(col)
    elif diferencia > 0.10:
        print("  Decisión: MANTENER")
        columnas_mantener.append(col)
    elif diferencia > 0.05:
        print("  Decisión: REVISAR CON CONTEXTO")
    else:
        print("  Decisión: WINSORIZACIÓN")
        columnas_winsorizacion.append(col)

# Eliminación de los outliers fuera de rango
for col in columnas_eliminar:
    rango_min, rango_max = rangos_validos[col]
    
    z = np.abs(scipy.stats.zscore(data[col].dropna()))
    mascara_z = z > 3
    mascara_rango = (data[col] < rango_min) | (data[col] > rango_max)
    mascara_eliminar = mascara_z & mascara_rango
    
    antes = len(data)
    data = data[~mascara_eliminar]
    print(f"{col}: {antes - len(data)} filas eliminadas.")


# Winsorización de los outliers correspondientes
for col in columnas_winsorizacion:
    data[col] = winsorize(data[col], limits=[0.01, 0.01])
    print(f"{col}: winsorización aplicada.")


# Análisis de la correlación entre variables
corr = data.corr(method="pearson")
print(corr)

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidth=0.5)
plt.show()

# Análisis de multicolinealidad entre las variables:
# Multicolinealidad (cálculo del VIF: factor de inflación de la varianza)
vif = pd.DataFrame()
vif["variable"] = data.columns
vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

print(vif)


# Visualización de las variables
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(columnas_con_outliers):
    axes[i].hist(data[col], bins=30, edgecolor='white')
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frecuencia')

plt.suptitle('Distribución de las variables predictoras', fontsize=16)
plt.tight_layout()
plt.show()

# Análisis del balanceo de clases en la variable objetivo 'Potability'
fig, ax = plt.subplots(figsize=(6, 4))
data['Potability'].value_counts().plot(kind='bar', ax=ax, edgecolor='white')
ax.set_title('Distribución de la variable objetivo (Potability)')
ax.set_xlabel('Potability')
ax.set_ylabel('Frecuencia')
ax.set_xticks([0, 1])
ax.set_xticklabels(['No potable (0)', 'Potable (1)'], rotation=0)
plt.tight_layout()
plt.show()

# Principales métricas
print(data.describe())

# División del dataset en train y test
X = data.drop(columns='Potability')
y = data['Potability'] # Variable objetivo

# Split entre train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

print(data.shape)
print(f"El nº de observaciones para el subconjunto de entrenamiento del dataset 'X' es de {X_train.shape[0]} observaciones.")
print(f"El nº de observaciones para el subconjunto de testeo del dataset 'X' es de {X_test.shape[0]} observaciones.\n")
print(f"El nº de observaciones para el subconjunto de entrenamiento de la variable objetivo 'y' es de {y_train.shape[0]} observaciones.")
print(f"El nº de observaciones para el subconjunto de testeo de la variable objetivo 'y' es de {y_test.shape[0]} observaciones.")

# Escalado
pipeline_escalado = Pipeline([('scaler', StandardScaler())])

# Fit solo en el subconjunto train, transform en ambos
X_train_escalado = pipeline_escalado.fit_transform(X_train)
X_test_escalado = pipeline_escalado.transform(X_test)

# Convertimos de vuelta a DataFrame para mantener los nombres de las columnas
X_train_escalado = pd.DataFrame(X_train_escalado, columns=X.columns)
X_test_escalado = pd.DataFrame(X_test_escalado, columns=X.columns)

print(X_train_escalado.describe().loc[['mean', 'std']].round(2))
print(X_train_escalado.head())

# Selección de variables
# Detecta relaciones lineales y no lineales
importancias = mutual_info_classif(X_train_escalado, y_train, random_state=42)
ranking_filtro = pd.Series(importancias, index=X_train.columns).sort_values(ascending=False)
print(ranking_filtro)

# Selección de variables usando Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_escalado, y_train)

importancias_rf = pd.Series(rf.feature_importances_, 
                             index=X_train.columns).sort_values(ascending=False)
print(importancias_rf)
