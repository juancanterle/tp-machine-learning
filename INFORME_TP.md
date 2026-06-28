# Trabajo Práctico Integrador – Machine Learning

**Materia:** Programación en Python
**Profesor:** Mg. Ing. Pedro D. López
**Cursada:** 2026

---

## 1. Objetivo

Desarrollar una aplicación en Python que, mediante un **menú de consola**, implemente
los distintos algoritmos de Machine Learning vistos en la playlist, aplicando en cada
caso un **dataset distinto** al usado en los ejemplos originales. Por cada algoritmo se
entrena un modelo, se muestran métricas, se generan gráficos básicos y se guarda un
informe con los resultados.

## 2. Descripción de la aplicación

El programa (`main.py`) presenta un menú con 12 algoritmos (6 de regresión y 6 de
clasificación), más una opción para **ejecutar todos** y la opción **0. Salir**.

Flujo de cada opción:

1. Se carga el dataset elegido (todos vienen incluidos en scikit-learn → funciona sin internet).
2. Se separan datos de entrenamiento y prueba (80/20) con semilla fija (`random_state=42`).
3. Se escalan las variables cuando el algoritmo lo requiere (SVR, SVM, KNN, Reg. Logística).
4. Se entrena el modelo y se predice sobre el set de prueba.
5. Se calculan métricas y se guardan gráficos en `informe/figuras/`.
6. Se vuelca todo a un informe en `informe/informe_resultados.md`.

**Métricas usadas:**
- *Regresión:* MAE, MSE, RMSE y R² (coeficiente de determinación).
- *Clasificación:* Accuracy, Precision, Recall y F1-score, más matriz de confusión.

## 3. ¿Por qué estos datasets?

La consigna pide datasets **distintos** a los de los ejemplos de la playlist. Se eligieron
los datasets embebidos en scikit-learn (no requieren descarga) y un dataset sintético,
de modo que la app sea reproducible en cualquier computadora:

| # | Algoritmo (video) | Dataset elegido |
|---|---|---|
| 1 | Regresión lineal simple (V16) | Diabetes (solo variable BMI) |
| 2 | Regresión lineal múltiple (V18) | California Housing *(respaldo: sintético multivariable)* |
| 3 | Regresión polinomial (V21) | Sintético no lineal (`y = 0.5x³ - x² + 2x + ruido`) |
| 4 | SVR – Vectores de soporte (V24) | Diabetes (10 variables) |
| 5 | Árbol de decisión – regresión (V27) | California Housing *(respaldo: sintético multivariable)* |
| 6 | Bosque aleatorio – regresión (V31) | Diabetes (10 variables) |
| 7 | Regresión logística (V39) | Breast Cancer |
| 8 | K vecinos más cercanos – KNN (V42) | Wine |
| 9 | SVM – clasificación (V46) | Digits |
| 10 | Naive Bayes (V49) | Wine |
| 11 | Árbol de decisión – clasificación (V52) | Breast Cancer |
| 12 | Bosque aleatorio – clasificación (V55) | Digits |

> Nota: California Housing se descarga la primera vez. Si no hay internet o fallan los
> certificados SSL (común en macOS), la app usa automáticamente un dataset **sintético
> multivariable** generado con `make_regression`, también distinto al del ejemplo.

## 4. Resultados obtenidos

### 4.1 Algoritmos de regresión (métrica clave: R², ideal → 1)

| # | Algoritmo | Dataset | MAE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | Lineal Simple | Diabetes (BMI) | 52.26 | 63.73 | 0.23 |
| 2 | Lineal Múltiple | Sintético (8 vars) | 11.32 | 14.48 | 0.99 |
| 3 | Polinomial (grado 3) | Sintético no lineal | 1.88 | 2.45 | 0.93 |
| 4 | SVR (kernel RBF) | Diabetes | 39.42 | 51.06 | 0.51 |
| 5 | Árbol de decisión | Sintético (8 vars) | 56.76 | 69.84 | 0.69 |
| 6 | Bosque aleatorio | Diabetes | 44.28 | 54.46 | 0.44 |

### 4.2 Algoritmos de clasificación (métrica clave: Accuracy, ideal → 1)

| # | Algoritmo | Dataset | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 7 | Regresión Logística | Breast Cancer | 0.98 | 0.98 | 0.98 | 0.98 |
| 8 | KNN (k=5) | Wine | 0.97 | 0.97 | 0.97 | 0.97 |
| 9 | SVM (kernel RBF) | Digits | 0.98 | 0.98 | 0.98 | 0.98 |
| 10 | Naive Bayes | Wine | 0.97 | 0.97 | 0.97 | 0.97 |
| 11 | Árbol de decisión | Breast Cancer | 0.92 | 0.92 | 0.92 | 0.92 |
| 12 | Bosque aleatorio | Digits | 0.96 | 0.96 | 0.96 | 0.96 |

*(Valores reales obtenidos al ejecutar el programa con semilla 42. El detalle completo y
los gráficos están en `informe/informe_resultados.md`.)*

## 5. Captura de resultados

Los gráficos se generan automáticamente en la carpeta `informe/figuras/`:

- **Regresión:** gráfico de recta/curva ajustada y dispersión "valor real vs predicho"
  (cuanto más cerca de la diagonal roja, mejor).
- **Clasificación:** matriz de confusión (diagonal = aciertos).

Archivos generados:
`reg_lineal_simple_recta.png`, `reg_lineal_simple_disp.png`, `reg_lineal_multiple_disp.png`,
`reg_polinomial_curva.png`, `svr_disp.png`, `arbol_reg_disp.png`, `bosque_reg_disp.png`,
`logistica_cm.png`, `knn_cm.png`, `svm_cm.png`, `nb_cm.png`, `arbol_clf_cm.png`,
`bosque_clf_cm.png`.

## 6. Comparación con lo esperado (conclusiones)

- **Regresión lineal simple vs múltiple:** con una sola variable (BMI) el R² es bajo
  (~0.23); al sumar todas las variables el modelo mejora notablemente. Confirma que más
  información relevante mejora la predicción.
- **Regresión polinomial:** con grado 3 ajusta muy bien (R² ~0.93) datos no lineales donde
  una recta fracasaría. Lo esperado.
- **SVR y modelos de árbol:** captan no linealidades; en datasets casi lineales (Diabetes)
  el R² queda en valores medios, coherente con la naturaleza de los datos.
- **Bosque vs árbol único:** el bosque reduce la varianza y suele igualar o superar al árbol
  individual, evitando sobreajuste (efecto de ensamble).
- **Clasificación:** todos los modelos superan el 92% de exactitud. Breast Cancer, Wine y
  Digits son datasets bien separables, por eso los resultados son altos, como se esperaba.
  El escalado de variables fue clave en KNN, SVM y Regresión Logística.

## 7. Cómo ejecutar

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Luego elegir una opción del menú (1–12), o la **13** para ejecutar todos los algoritmos
y generar el informe completo. La opción **0** sale del programa.
