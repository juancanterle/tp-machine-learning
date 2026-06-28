# TP Integrador – Machine Learning

Aplicación de consola en Python que implementa **12 algoritmos de Machine Learning**
(6 de regresión y 6 de clasificación) mediante un menú interactivo. Cada algoritmo usa
un dataset distinto al de los ejemplos de la playlist, entrena un modelo, muestra
métricas, genera gráficos y guarda un informe.

**Materia:** Programación en Python · **Profesor:** Mg. Ing. Pedro D. López · **Cursada:** 2026

## Algoritmos implementados

| # | Algoritmo | Tipo | Dataset |
|---|---|---|---|
| 1 | Regresión Lineal Simple | Regresión | Diabetes (BMI) |
| 2 | Regresión Lineal Múltiple | Regresión | California Housing / Sintético |
| 3 | Regresión Polinomial | Regresión | Sintético no lineal |
| 4 | SVR (Vectores de Soporte) | Regresión | Diabetes |
| 5 | Árbol de Decisión | Regresión | California Housing / Sintético |
| 6 | Bosque Aleatorio | Regresión | Diabetes |
| 7 | Regresión Logística | Clasificación | Breast Cancer |
| 8 | K Vecinos Más Cercanos (KNN) | Clasificación | Wine |
| 9 | SVM | Clasificación | Digits |
| 10 | Naive Bayes | Clasificación | Wine |
| 11 | Árbol de Decisión | Clasificación | Breast Cancer |
| 12 | Bosque Aleatorio | Clasificación | Digits |

## Cómo ejecutar

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
python main.py
```

En el menú: elegí una opción **1–12**, o la **13** para correr todos los algoritmos y
generar el informe completo. La opción **0** sale.

## Salidas

- `informe/informe_resultados.md` — informe autogenerado con métricas y gráficos.
- `informe/figuras/` — gráficos en PNG (dispersión real vs predicho, curvas, matrices de confusión).
- `INFORME_TP.md` — informe escrito explicativo (entregable).

## Tecnologías

Python · scikit-learn · NumPy · pandas · matplotlib
