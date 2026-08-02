# Water Potability - Análisis y modelado predictivo
Este proyecto analiza el dataset Water Potability con el objetivo de determinar si una muestra de agua es potable o no mediante técnicas de análisis de datos y modelado.

El trabajo incluye preprocesado, análisis exploratorio y construcción de modelos predictivos, siguiendo un pipeline completo de Data Science.

## Objetivos del proyecto
Construir un modelo capaz de predecir la potabilidad del agua a partir de variables químicas como:

* pH
* Dureza
* Sólidos
* Cloraminas
* Sulfatos
* Conductividad
* Carbón orgánico
* Trihalometanos
* Turbidez

El objetivo no es maximizar la precisión global (accuracy), sino identificar correctamente las muestras de agua no potable. En este contexto, el error más crítico es un **falso negativo**, es decir, clasificar agua no potable como potable, ya que esto puede suponer un riesgo grave para la salud.

Por ello, se pone especial énfasis en el **recall de la clase "no potable"**, asegurando que el agua contaminada sea detectada de la forma más fiable posible, incluso a costa de aumentar el número de falsos positivos. Este enfoque prioriza la seguridad frente a métricas generales como el accuracy.

## Dataset
* **Nombre:** Water Potability
* **Número de observaciones:** 3276
* **Número de variables:** 10
* **Variable objetivo:** Potability (0 = No potable, 1 = Potable)

El dataset recoge características químicas del agua que influyen directamente en su calidad y consumo humano.

**Nota:** El dataset está incluido en este repositorio (`water_potability.csv`) para garantizar la reproducibilidad del proyecto.

## Preprocesado de los datos
* Estudio de valores faltantes (NA's).
* Estudio de valores atípicos.
* Análisis de correlaciones entre variables.
* Análisis de multicolinealidad (factor de inflación de la varianza).
* Visualización de las variables.
* Estudio de las principales métricas.
* Escalado/normalización.
* División del dataset en los subconjuntos de train y test.
* Selección de variables.

## Modelado
Se aplican diversos modelos de Machine Learning para clasificar la potabilidad del agua, en concreto se usan:
* Regresión logística sin transformación.
* Regresión logística con transformación Yeo-Johnson.
* Regresión logística con penalización Lasso.
* Regresión logística con penalización Ridge.
* Árbol de decisión.
* Random Forest.
* Gradient Boosting.
* XGBoost.

## Métricas de evaluación
Se han utilizado las siguientes métricas:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Resultados
En comparación con un modelo base aleatorio (AUC ≈ 0.5), las regresiones logísticas no muestran capacidad predictiva real, ya que se mantienen en valores cercanos al azar. Los modelos basados en árboles mejoran claramente este baseline:

* **Random Forest, Gradient Boosting y XGBoost** alcanzan AUC ≈ 0.66–0.68, indicando cierta capacidad de discriminación.
* **XGBoost** destaca como el mejor modelo global, obteniendo el mayor F1-score (~0.54) y el mejor recall (~0.55).

Dado que el objetivo del problema es evitar falsos negativos (clasificar agua no potable como potable), el recall es la métrica más relevante. En este sentido, **XGBoost resulta el modelo más adecuado al maximizar la detección de agua no potable**.


## Conclusiones
* La potabilidad del agua no depende de una única variable, sino de la combinación de varias características químicas.
* El tratamiento de valores faltantes es crítico para el rendimiento del modelo.
* Los modelos no lineales capturan mejor estas relaciones complejas.


## Tecnologías utilizadas
* Python
* Pandas
* NumPy
* Matplotlib/Seaborn
* Scikit-learn
* Jupyter Notebook

## Estructura del proyecto

potabilidad_agua/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── water_potability.csv
│
├── notebooks/
│   └── 
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│
├── results/
│   ├── comparacion_modelos.png
│   └── resultados_numericos.jpg

## Contexto
El acceso a agua potable es un problema crítico a nivel global. Este proyecto muestra cómo técnicas de **Data Science** puede contribuir a:
* Evaluar la calidad del agua.
* Automatizar diagnósticos.
* Apoyar decisiones fundamentales en salud pública.

## Reproducibilidad

El proyecto es completamente reproducible:

1. Clonar el repositorio
2. Instalar dependencias con `requirements.txt`
3. Ejecutar el notebook

El dataset está incluido en `data/`.

## Autor
**Pablo De la Fuente Arteaga**
* Matemáticas
* Data Science
* Python | Machine Learning | Deep Learning
