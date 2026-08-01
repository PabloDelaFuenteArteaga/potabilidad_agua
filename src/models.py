from preprocessing import (
    X, y,
    X_train_escalado, y_train,
    X_test_escalado, y_test
)
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.metrics import RocCurveDisplay


# Pipeline del modelo de regresión logística sin penalización
pipeline_reg_logist_base = Pipeline([
    ("modelo", LogisticRegression(penalty=None, max_iter=1000, class_weight="balanced"))
])
pipeline_reg_logist_base.fit(X_train_escalado, y_train)
modelo_reg_logist_base_sm = Logit(y_train.values, add_constant(X_train_escalado)).fit(disp=0)

# Pipeline del modelo de regresión logística con transformación Yeo-Johnson
pipeline_reg_logist_yj = Pipeline([
    ("yeo_johnson", PowerTransformer(method="yeo-johnson")),
    ("modelo",      LogisticRegression(penalty=None, max_iter=1000, class_weight="balanced"))
])
pipeline_reg_logist_yj.fit(X_train_escalado, y_train)

# Extraemos X_train_yj del pipeline
yj_step    = pipeline_reg_logist_yj.named_steps["yeo_johnson"]
X_train_yj = pd.DataFrame(
    yj_step.transform(X_train_escalado),
    columns=X_train_escalado.columns
)

modelo_reg_logist_yj_sm = Logit(y_train.values, add_constant(X_train_yj)).fit(disp=0)
print(f"p-valor modelo base: {modelo_reg_logist_base_sm.llr_pvalue:.4f}")
print(f"p-valor modelo con Yeo-Johnson: {modelo_reg_logist_yj_sm.llr_pvalue:.4f}")


pipeline_reg_logist_lasso = Pipeline([
    ("yeo_john", PowerTransformer(method="yeo-johnson")),
    ("modelo_reg_logistic_lasso", LogisticRegression(penalty="l1", # Usamos L1 (Lasso)
                                                     solver="liblinear",
                                                     class_weight='balanced'))
])

grid_parametros_reg_logist_lasso = {"modelo_reg_logistic_lasso__C": 
                                    [0.01, 0.02, 0.05, 0.1, 0.5, 
                                    1, 2, 5, 10, 20, 50, 100]}

grid_reg_logist_lasso = GridSearchCV(
    pipeline_reg_logist_lasso,
    grid_parametros_reg_logist_lasso,
    cv=10,
    scoring="roc_auc",
    n_jobs=-1
)

grid_reg_logist_lasso.fit(X_train_escalado, y_train)
print(grid_reg_logist_lasso.best_params_)
mejor_modelo_reg_logist_lasso = grid_reg_logist_lasso.best_estimator_

# Coeficientes Lasso
coef_lasso = mejor_modelo_reg_logist_lasso.named_steps["modelo_reg_logistic_lasso"].coef_[0]
print(pd.Series(coef_lasso, index=X.columns).sort_values())


pipeline_reg_logist_ridge = Pipeline([
    ("yeo_john", PowerTransformer(method="yeo-johnson")),
    ("modelo_reg_logistic_ridge", LogisticRegression(penalty="l2", # Usamos L2 (Ridge)
                                                     solver="liblinear",
                                                     class_weight='balanced'))
])

grid_parametros_reg_logist_ridge = {"modelo_reg_logistic_ridge__C": 
                                    [0.01, 0.02, 0.05, 0.1, 0.5, 
                                    1, 2, 5, 10, 20, 50, 100]}

grid_reg_logist_ridge = GridSearchCV(
    pipeline_reg_logist_ridge,
    grid_parametros_reg_logist_ridge,
    cv=10,
    scoring="roc_auc",
    n_jobs=-1
)

grid_reg_logist_ridge.fit(X_train_escalado, y_train)
print(grid_reg_logist_ridge.best_params_)
mejor_modelo_reg_logist_ridge = grid_reg_logist_ridge.best_estimator_

# Coeficientes Ridge
coef_ridge = mejor_modelo_reg_logist_ridge.named_steps["modelo_reg_logistic_ridge"].coef_[0]
pd.Series(coef_ridge, index=X.columns).sort_values()


# Árbol de decisión
max_depth = [3, 5, 7, 10, 20, None]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

pipeline_arbol_decision = Pipeline([
    ("modelo_arbol_decision", DecisionTreeClassifier(random_state=0, class_weight='balanced'))
])

grid_parametros_arbol_decision = {"modelo_arbol_decision__max_depth": max_depth,
                                  "modelo_arbol_decision__min_samples_split": min_samples_split,
                                  "modelo_arbol_decision__min_samples_leaf": min_samples_leaf}

grid_arbol_decision = GridSearchCV(pipeline_arbol_decision,
                                   grid_parametros_arbol_decision,
                                   cv=10,
                                   scoring="roc_auc",
                                   n_jobs=-1)

grid_arbol_decision.fit(X_train_escalado, y_train)
print(grid_arbol_decision.best_params_)

mejor_modelo_arbol_decision = grid_arbol_decision.best_estimator_

# Análisis de posible sobreajuste del modelo
y_prob_arbol_decision_train = mejor_modelo_arbol_decision.predict_proba(X_train_escalado)[:, 1]
y_prob_arbol_decision_test  = mejor_modelo_arbol_decision.predict_proba(X_test_escalado)[:, 1]

print(f"AUC-ROC train : {roc_auc_score(y_train, y_prob_arbol_decision_train):.4f}")
print(f"AUC-ROC test  : {roc_auc_score(y_test,  y_prob_arbol_decision_test):.4f}")


# Random Forest
pipeline_random_forest = Pipeline([
    ("modelo_random_forest", RandomForestClassifier(random_state=0, class_weight='balanced'))
])

grid_parametros_random_forest = {"modelo_random_forest__max_depth": max_depth,
                                 "modelo_random_forest__min_samples_split": min_samples_split,
                                 "modelo_random_forest__min_samples_leaf": min_samples_leaf}

grid_random_forest = GridSearchCV(pipeline_random_forest,
                                  grid_parametros_random_forest,
                                  cv=10,
                                  scoring="roc_auc",
                                  n_jobs=-1)

grid_random_forest.fit(X_train_escalado, y_train)
print(grid_random_forest.best_params_)

mejor_modelo_random_forest = grid_random_forest.best_estimator_

# Análisis de posible sobreajuste
y_prob_random_forest_train = mejor_modelo_random_forest.predict_proba(X_train_escalado)[:, 1]
y_prob_random_forest_test  = mejor_modelo_random_forest.predict_proba(X_test_escalado)[:, 1]
print(f"AUC-ROC train : {roc_auc_score(y_train, y_prob_random_forest_train):.4f}")
print(f"AUC-ROC test  : {roc_auc_score(y_test,  y_prob_random_forest_test):.4f}")


# Gradient Boosting
pesos = compute_sample_weight(class_weight='balanced', y=y_train)

pipeline_gradient_boosting = Pipeline([
    ("modelo_gradient_boosting", GradientBoostingClassifier(random_state=0))
])

grid_parametros_gradient_boosting = {"modelo_gradient_boosting__max_depth": max_depth,
                                 "modelo_gradient_boosting__min_samples_split": min_samples_split,
                                 "modelo_gradient_boosting__min_samples_leaf": min_samples_leaf}

grid_gradient_boosting = GridSearchCV(pipeline_gradient_boosting,
                                  grid_parametros_gradient_boosting,
                                  cv=10,
                                  scoring="roc_auc",
                                  n_jobs=-1)

grid_gradient_boosting.fit(X_train_escalado, y_train, modelo_gradient_boosting__sample_weight=pesos)
print(grid_gradient_boosting.best_params_)

mejor_modelo_gradient_boosting = grid_gradient_boosting.best_estimator_

# Análisis del posible sobreajuste
y_prob_gradient_boosting_train = mejor_modelo_gradient_boosting.predict_proba(X_train_escalado)[:, 1]
y_prob_gradient_boosting_test  = mejor_modelo_gradient_boosting.predict_proba(X_test_escalado)[:, 1]

print(f"AUC-ROC train : {roc_auc_score(y_train, y_prob_gradient_boosting_train):.4f}")
print(f"AUC-ROC test  : {roc_auc_score(y_test,  y_prob_gradient_boosting_test):.4f}")


# Creamos pesos para corregir el desbalanceo de clases
scale = (y_train == 0).sum() / (y_train == 1).sum()

pipeline_xgboost = Pipeline([
    ("modelo_xgboost", XGBClassifier(random_state=0,
                                     eval_metric="auc",
                                     verbosity=0,
                                     scale_pos_weight=scale))
])
grid_parametros_xgboost = {"modelo_xgboost__n_estimators": [100, 200, 300],
                           "modelo_xgboost__max_depth": [3, 5, 7],
                           "modelo_xgboost__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9]}

grid_xgboost = GridSearchCV(pipeline_xgboost,
                                  grid_parametros_xgboost,
                                  cv=10,
                                  scoring="roc_auc",
                                  n_jobs=-1)

grid_xgboost.fit(X_train_escalado, y_train)
print(grid_xgboost.best_params_)

mejor_modelo_xgboost = grid_xgboost.best_estimator_

y_prob_xgboost_train = mejor_modelo_xgboost.predict_proba(X_train_escalado)[:, 1]
y_prob_xgboost_test  = mejor_modelo_xgboost.predict_proba(X_test_escalado)[:, 1]

print(f"AUC-ROC train : {roc_auc_score(y_train, y_prob_xgboost_train):.4f}")
print(f"AUC-ROC test  : {roc_auc_score(y_test,  y_prob_xgboost_test):.4f}")


def evaluacion_modelo(nombre_modelo, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)


    dicc_metricas = {"Nombre modelo": nombre_modelo,
                     "Accuracy": accuracy,
                     "AUC-ROC": auc_roc,
                     "F1-score": f1,
                     "Precisión": precision,
                     "Recall": recall}
    
    return dicc_metricas


resultados = []

resultados.append(evaluacion_modelo("Regresión Logística", 
                  pipeline_reg_logist_base, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("Regresión Logística con Yeo-Johnson",
                  pipeline_reg_logist_yj, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("Regresión Logística Lasso", 
                  mejor_modelo_reg_logist_lasso, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("Regresión Logística Ridge", 
                  mejor_modelo_reg_logist_ridge, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("Árbol de decisión", 
                  mejor_modelo_arbol_decision, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("Random Forest", 
                  mejor_modelo_random_forest, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("Gradient Boosting", 
                  mejor_modelo_gradient_boosting, X_test_escalado, y_test))
resultados.append(evaluacion_modelo("XGBoost", 
                  mejor_modelo_xgboost, X_test_escalado, y_test))

dataframe = pd.DataFrame(resultados)
print(dataframe)

modelos_finales = {
    'Random Forest':      mejor_modelo_random_forest,
    'Gradient Boosting':  mejor_modelo_gradient_boosting,
    'XGBoost':            mejor_modelo_xgboost
}

for nombre, modelo in modelos_finales.items():
    y_pred = modelo.predict(X_test_escalado)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No potable', 'Potable'])
    disp.plot()
    plt.title(f'Matriz de confusión — {nombre}')
    plt.show()


fig, ax = plt.subplots(figsize=(8, 6))

for nombre, modelo in modelos_finales.items():
    RocCurveDisplay.from_estimator(modelo, X_test_escalado, y_test,
                                    ax=ax, name=nombre)

ax.plot([0, 1], [0, 1], 'k--', label='Azar')
ax.set_title('Curva ROC — comparación de modelos')
plt.show()
