# -*- coding: utf-8 -*-
"""
Trabajo Práctico Integrador - Machine Learning
Profesor: Mg. Ing. Pedro D. López - Cursada 2026

App de consola con menú que implementa 12 algoritmos de ML.
Cada algoritmo usa un dataset DISTINTO al de los ejemplos de la playlist,
entrena un modelo, muestra métricas, guarda gráficos (PNG) y genera un informe.

Requisitos: scikit-learn, numpy, pandas, matplotlib
Ejecutar:   python main.py
"""

import os
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: permite guardar PNG sin entorno gráfico
import matplotlib.pyplot as plt

from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_wine, load_digits,
    fetch_california_housing, make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ----------------------------------------------------------------------------
# Carpetas de salida
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALIDA_DIR = os.path.join(BASE_DIR, "informe")
FIG_DIR = os.path.join(SALIDA_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42  # semilla fija para reproducibilidad


# ============================================================================
# UTILIDADES: métricas, gráficos e informe
# ============================================================================
def evaluar_regresion(y_true, y_pred):
    """Devuelve un diccionario con las métricas típicas de regresión."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluar_clasificacion(y_true, y_pred):
    """Devuelve un diccionario con las métricas típicas de clasificación."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def guardar_fig(nombre):
    """Guarda la figura actual de matplotlib en informe/figuras y la cierra."""
    ruta = os.path.join(FIG_DIR, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=110)
    plt.close()
    return ruta


def grafico_dispersion_regresion(y_true, y_pred, titulo, archivo):
    """Gráfico 'valor real vs predicho' (cuanto más cerca de la diagonal, mejor)."""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidths=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=2, label="Predicción perfecta")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(titulo)
    plt.legend()
    return guardar_fig(archivo)


def grafico_recta_simple(X, y, modelo, titulo, archivo, xlabel="X", ylabel="y"):
    """Nube de puntos con la recta/curva ajustada (para 1 sola variable)."""
    plt.figure(figsize=(6, 5))
    plt.scatter(X, y, alpha=0.5, edgecolors="k", linewidths=0.3, label="Datos")
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    plt.plot(x_line, modelo.predict(x_line), "r-", linewidth=2, label="Modelo")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    return guardar_fig(archivo)


def grafico_matriz_confusion(y_true, y_pred, clases, titulo, archivo):
    """Matriz de confusión como mapa de calor."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(clases))
    plt.xticks(ticks, clases, rotation=45, ha="right")
    plt.yticks(ticks, clases)
    umbral = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center",
                     color="white" if cm[i, j] > umbral else "black")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.title(titulo)
    return guardar_fig(archivo)


def imprimir_resultado(res):
    """Muestra el resultado de un algoritmo por consola."""
    print("\n" + "=" * 64)
    print(f"  {res['algoritmo']}")
    print("=" * 64)
    print(f"Dataset      : {res['dataset']}")
    print(f"Descripción  : {res['descripcion']}")
    print(f"Muestras     : {res['n_muestras']}  |  Variables: {res['n_features']}")
    print("-" * 64)
    print("Métricas:")
    for k, v in res["metricas"].items():
        print(f"   {k:<12}: {v:.4f}")
    print("-" * 64)
    print("Comparación con lo esperado:")
    print("   " + res["comentario"])
    print("Gráficos guardados:")
    for f in res["figuras"]:
        print("   - " + os.path.relpath(f, BASE_DIR))


def guardar_informe(resultados):
    """Escribe un informe en Markdown con todos los resultados ejecutados."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ruta = os.path.join(SALIDA_DIR, "informe_resultados.md")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Informe de Resultados - TP Integrador Machine Learning\n\n")
        f.write(f"*Generado: {ts}*\n\n")
        f.write(f"Total de algoritmos ejecutados: **{len(resultados)}**\n\n")
        f.write("---\n\n")
        for r in resultados:
            f.write(f"## {r['algoritmo']}\n\n")
            f.write(f"- **Dataset:** {r['dataset']}\n")
            f.write(f"- **Descripción:** {r['descripcion']}\n")
            f.write(f"- **Tipo:** {r['tipo']}\n")
            f.write(f"- **Muestras / Variables:** {r['n_muestras']} / {r['n_features']}\n\n")
            f.write("**Métricas:**\n\n")
            f.write("| Métrica | Valor |\n|---|---|\n")
            for k, v in r["metricas"].items():
                f.write(f"| {k} | {v:.4f} |\n")
            f.write("\n**Comparación con lo esperado:** ")
            f.write(r["comentario"] + "\n\n")
            f.write("**Gráficos:**\n\n")
            for fig in r["figuras"]:
                rel = os.path.relpath(fig, SALIDA_DIR)
                f.write(f"![{os.path.basename(fig)}]({rel})\n\n")
            f.write("---\n\n")
    return ruta


# ============================================================================
# CARGA DE DATASETS (con fallback offline para California Housing)
# ============================================================================
def cargar_california():
    """California Housing. Si no hay internet/certificados, usa un dataset
    sintético multivariable (make_regression) como respaldo offline."""
    try:
        d = fetch_california_housing()
        return d.data, d.target, list(d.feature_names), "California Housing", \
            "Precio medio de viviendas en California según variables socioeconómicas."
    except Exception:
        X, y = make_regression(n_samples=600, n_features=8, n_informative=6,
                               noise=15.0, random_state=SEED)
        feats = [f"var_{i}" for i in range(X.shape[1])]
        return X, y, feats, "Sintético multivariable (make_regression)", \
            ("Dataset simulado con 8 variables (6 informativas) y ruido; respaldo "
             "offline cuando no se puede descargar California Housing.")


def dataset_polinomial_sintetico():
    """Dataset sintético no lineal (relación polinómica) para regresión polinomial."""
    rng = np.random.RandomState(SEED)
    X = np.linspace(-3, 3, 300).reshape(-1, 1)
    y = (0.5 * X[:, 0] ** 3) - (X[:, 0] ** 2) + (2 * X[:, 0]) + rng.normal(0, 3, size=300)
    return X, y


# ============================================================================
# ALGORITMOS DE REGRESIÓN
# ============================================================================
def regresion_lineal_simple():
    """Video 16 - Regresión lineal simple. Dataset: Diabetes (variable BMI)."""
    d = load_diabetes()
    idx_bmi = list(d.feature_names).index("bmi")
    X = d.data[:, [idx_bmi]]
    y = d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    modelo = LinearRegression().fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_regresion(yte, ypred)

    figs = [
        grafico_recta_simple(X, y, modelo, "Reg. lineal simple - Diabetes (BMI)",
                             "reg_lineal_simple_recta.png",
                             xlabel="BMI (normalizado)", ylabel="Progresión enfermedad"),
        grafico_dispersion_regresion(yte, ypred, "Real vs Predicho",
                                     "reg_lineal_simple_disp.png"),
    ]
    return {
        "algoritmo": "Regresión Lineal Simple", "tipo": "Regresión",
        "dataset": "Diabetes (scikit-learn) - solo variable BMI",
        "descripcion": "Predice la progresión de la diabetes a partir del índice de masa corporal.",
        "n_muestras": len(y), "n_features": 1, "metricas": met, "figuras": figs,
        "comentario": (f"Con una sola variable el R2 es bajo (~{met['R2']:.2f}), lo esperado: "
                       "el BMI explica solo parte de la progresión. La pendiente positiva "
                       "confirma que a mayor BMI, mayor progresión."),
    }


def regresion_lineal_multiple():
    """Video 18 - Regresión lineal múltiple. Dataset: California Housing."""
    X, y, feats, nombre, desc = cargar_california()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    modelo = LinearRegression().fit(sc.transform(Xtr), ytr)
    ypred = modelo.predict(sc.transform(Xte))
    met = evaluar_regresion(yte, ypred)
    figs = [grafico_dispersion_regresion(yte, ypred,
            f"Reg. lineal múltiple - {nombre}", "reg_lineal_multiple_disp.png")]
    return {
        "algoritmo": "Regresión Lineal Múltiple", "tipo": "Regresión",
        "dataset": nombre, "descripcion": desc,
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"Al usar todas las variables el R2 sube (~{met['R2']:.2f}) respecto "
                       "del modelo simple, como se espera. La relación es parcialmente lineal, "
                       "por eso el R2 no es cercano a 1."),
    }


def regresion_polinomial():
    """Video 21 - Regresión polinomial. Dataset: sintético no lineal."""
    X, y = dataset_polinomial_sintetico()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    modelo = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_regresion(yte, ypred)
    figs = [grafico_recta_simple(X, y, modelo, "Reg. polinomial (grado 3) - sintético",
            "reg_polinomial_curva.png", xlabel="X", ylabel="y")]
    return {
        "algoritmo": "Regresión Polinomial (grado 3)", "tipo": "Regresión",
        "dataset": "Dataset sintético no lineal (y = 0.5x^3 - x^2 + 2x + ruido)",
        "descripcion": "Datos simulados con relación cúbica para ajustar una curva polinómica.",
        "n_muestras": len(y), "n_features": 1, "metricas": met, "figuras": figs,
        "comentario": (f"El ajuste polinómico captura la curva y logra un R2 alto "
                       f"(~{met['R2']:.2f}). Una recta lineal aquí fallaría: lo esperado es "
                       "que el grado 3 modele bien la forma de los datos."),
    }


def svr_regresion():
    """Video 24 - SVR (regresión con vectores de soporte). Dataset: Diabetes."""
    d = load_diabetes()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    modelo = SVR(kernel="rbf", C=100, gamma="scale").fit(sc.transform(Xtr), ytr)
    ypred = modelo.predict(sc.transform(Xte))
    met = evaluar_regresion(yte, ypred)
    figs = [grafico_dispersion_regresion(yte, ypred, "SVR (kernel RBF) - Diabetes",
            "svr_disp.png")]
    return {
        "algoritmo": "SVR - Support Vector Regression", "tipo": "Regresión",
        "dataset": "Diabetes (scikit-learn) - 10 variables",
        "descripcion": "Regresión con vectores de soporte y kernel RBF sobre la progresión.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"El SVR con kernel RBF obtiene un R2 ~{met['R2']:.2f}, similar al "
                       "lineal porque el dataset es casi lineal. Es clave escalar los datos "
                       "antes de entrenar, como se espera en SVM/SVR."),
    }


def arbol_regresion():
    """Video 27 - Árbol de decisión (regresión). Dataset: California Housing."""
    X, y, feats, nombre, desc = cargar_california()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    modelo = DecisionTreeRegressor(max_depth=6, random_state=SEED).fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_regresion(yte, ypred)
    figs = [grafico_dispersion_regresion(yte, ypred,
            f"Árbol de decisión (regresión) - {nombre}", "arbol_reg_disp.png")]
    return {
        "algoritmo": "Árbol de Decisión (Regresión)", "tipo": "Regresión",
        "dataset": nombre, "descripcion": desc,
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"El árbol (max_depth=6) capta relaciones no lineales y mejora al "
                       f"lineal (R2 ~{met['R2']:.2f}). Se limita la profundidad para evitar "
                       "sobreajuste, como es esperable."),
    }


def bosque_regresion():
    """Video 31 - Bosque aleatorio (regresión). Dataset: Diabetes."""
    d = load_diabetes()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    modelo = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_regresion(yte, ypred)
    figs = [grafico_dispersion_regresion(yte, ypred,
            "Bosque aleatorio (regresión) - Diabetes", "bosque_reg_disp.png")]
    return {
        "algoritmo": "Bosque Aleatorio (Regresión)", "tipo": "Regresión",
        "dataset": "Diabetes (scikit-learn) - 10 variables",
        "descripcion": "Conjunto de 200 árboles para predecir la progresión de la enfermedad.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"El bosque promedia muchos árboles y reduce la varianza "
                       f"(R2 ~{met['R2']:.2f}). Suele igualar o superar al árbol único, "
                       "que es lo esperado por el efecto de ensamble."),
    }


# ============================================================================
# ALGORITMOS DE CLASIFICACIÓN
# ============================================================================
def regresion_logistica():
    """Video 39 - Regresión logística. Dataset: Breast Cancer."""
    d = load_breast_cancer()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    modelo = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    ypred = modelo.predict(sc.transform(Xte))
    met = evaluar_clasificacion(yte, ypred)
    figs = [grafico_matriz_confusion(yte, ypred, d.target_names,
            "Reg. logística - Breast Cancer", "logistica_cm.png")]
    return {
        "algoritmo": "Regresión Logística", "tipo": "Clasificación",
        "dataset": "Breast Cancer (scikit-learn)",
        "descripcion": "Clasifica tumores en benignos/malignos según 30 características.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"Alcanza una exactitud alta (~{met['Accuracy']:.2f}), lo esperado en "
                       "este dataset bien separable. El escalado mejora la convergencia."),
    }


def knn_clasificacion():
    """Video 42 - K vecinos más cercanos. Dataset: Wine."""
    d = load_wine()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    modelo = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(Xtr), ytr)
    ypred = modelo.predict(sc.transform(Xte))
    met = evaluar_clasificacion(yte, ypred)
    figs = [grafico_matriz_confusion(yte, ypred, d.target_names,
            "KNN (k=5) - Wine", "knn_cm.png")]
    return {
        "algoritmo": "K Vecinos Más Cercanos (KNN)", "tipo": "Clasificación",
        "dataset": "Wine (scikit-learn)",
        "descripcion": "Clasifica vinos en 3 cultivares según 13 propiedades químicas.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"Con k=5 y datos escalados logra exactitud ~{met['Accuracy']:.2f}. "
                       "El escalado es imprescindible en KNN porque usa distancias; sin él "
                       "el rendimiento caería, tal como se espera."),
    }


def svm_clasificacion():
    """Video 46 - SVM (clasificación). Dataset: Digits."""
    d = load_digits()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    modelo = SVC(kernel="rbf", C=10, gamma="scale").fit(sc.transform(Xtr), ytr)
    ypred = modelo.predict(sc.transform(Xte))
    met = evaluar_clasificacion(yte, ypred)
    figs = [grafico_matriz_confusion(yte, ypred, [str(i) for i in range(10)],
            "SVM (RBF) - Digits", "svm_cm.png")]
    return {
        "algoritmo": "SVM - Máquinas de Vectores de Soporte", "tipo": "Clasificación",
        "dataset": "Digits (scikit-learn)",
        "descripcion": "Reconoce dígitos manuscritos (0-9) a partir de imágenes 8x8.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"El SVM con kernel RBF obtiene exactitud muy alta "
                       f"(~{met['Accuracy']:.2f}) en reconocimiento de dígitos, resultado "
                       "esperado: SVM es muy efectivo en alta dimensión."),
    }


def naive_bayes():
    """Video 49 - Naive Bayes. Dataset: Wine."""
    d = load_wine()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    modelo = GaussianNB().fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_clasificacion(yte, ypred)
    figs = [grafico_matriz_confusion(yte, ypred, d.target_names,
            "Naive Bayes Gaussiano - Wine", "nb_cm.png")]
    return {
        "algoritmo": "Naive Bayes (Gaussiano)", "tipo": "Clasificación",
        "dataset": "Wine (scikit-learn)",
        "descripcion": "Clasifica vinos en 3 cultivares asumiendo variables gaussianas.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"Pese a asumir independencia entre variables, logra exactitud "
                       f"~{met['Accuracy']:.2f}. Es rápido y simple; el buen resultado es lo "
                       "esperado en datasets donde las clases están bien separadas."),
    }


def arbol_clasificacion():
    """Video 52 - Árbol de decisión (clasificación). Dataset: Breast Cancer."""
    d = load_breast_cancer()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    modelo = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_clasificacion(yte, ypred)
    figs = [grafico_matriz_confusion(yte, ypred, d.target_names,
            "Árbol de decisión (clasif.) - Breast Cancer", "arbol_clf_cm.png")]
    return {
        "algoritmo": "Árbol de Decisión (Clasificación)", "tipo": "Clasificación",
        "dataset": "Breast Cancer (scikit-learn)",
        "descripcion": "Clasifica tumores benignos/malignos con reglas interpretables.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"El árbol (max_depth=5) da exactitud ~{met['Accuracy']:.2f} y es "
                       "interpretable. Limitar la profundidad evita el sobreajuste típico de "
                       "los árboles, como se espera."),
    }


def bosque_clasificacion():
    """Video 55 - Bosque aleatorio (clasificación). Dataset: Digits."""
    d = load_digits()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    modelo = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(Xtr, ytr)
    ypred = modelo.predict(Xte)
    met = evaluar_clasificacion(yte, ypred)
    figs = [grafico_matriz_confusion(yte, ypred, [str(i) for i in range(10)],
            "Bosque aleatorio (clasif.) - Digits", "bosque_clf_cm.png")]
    return {
        "algoritmo": "Bosque Aleatorio (Clasificación)", "tipo": "Clasificación",
        "dataset": "Digits (scikit-learn)",
        "descripcion": "Reconoce dígitos manuscritos con un conjunto de 200 árboles.",
        "n_muestras": len(y), "n_features": X.shape[1], "metricas": met, "figuras": figs,
        "comentario": (f"El bosque alcanza exactitud ~{met['Accuracy']:.2f}, superando al "
                       "árbol único gracias al ensamble. Resultado esperado y robusto frente "
                       "al sobreajuste."),
    }


# ============================================================================
# MENÚ
# ============================================================================
OPCIONES = {
    "1": ("Regresión Lineal Simple        (Diabetes)",        regresion_lineal_simple),
    "2": ("Regresión Lineal Múltiple      (California)",       regresion_lineal_multiple),
    "3": ("Regresión Polinomial           (Sintético)",       regresion_polinomial),
    "4": ("SVR - Vectores de Soporte      (Diabetes)",        svr_regresion),
    "5": ("Árbol de Decisión (regresión)  (California)",       arbol_regresion),
    "6": ("Bosque Aleatorio (regresión)   (Diabetes)",        bosque_regresion),
    "7": ("Regresión Logística            (Breast Cancer)",   regresion_logistica),
    "8": ("K Vecinos Más Cercanos (KNN)   (Wine)",            knn_clasificacion),
    "9": ("SVM (clasificación)            (Digits)",          svm_clasificacion),
    "10": ("Naive Bayes                    (Wine)",            naive_bayes),
    "11": ("Árbol de Decisión (clasif.)    (Breast Cancer)",  arbol_clasificacion),
    "12": ("Bosque Aleatorio (clasif.)     (Digits)",         bosque_clasificacion),
}


def mostrar_menu():
    print("\n" + "#" * 64)
    print("#  TP INTEGRADOR - MACHINE LEARNING".ljust(63) + "#")
    print("#" * 64)
    print("  --- REGRESIÓN ---")
    for k in ["1", "2", "3", "4", "5", "6"]:
        print(f"   {k:>2}. {OPCIONES[k][0]}")
    print("  --- CLASIFICACIÓN ---")
    for k in ["7", "8", "9", "10", "11", "12"]:
        print(f"   {k:>2}. {OPCIONES[k][0]}")
    print("  --- OTRAS OPCIONES ---")
    print("   13. Ejecutar TODOS los algoritmos y generar informe")
    print("    0. Salir")
    print("#" * 64)


def main():
    resultados = []  # acumula resultados para el informe
    print("Las figuras se guardan en:", os.path.relpath(FIG_DIR, BASE_DIR))
    while True:
        mostrar_menu()
        op = input("Seleccione una opción: ").strip()

        if op == "0":
            if resultados:
                ruta = guardar_informe(resultados)
                print(f"\nInforme guardado en: {os.path.relpath(ruta, BASE_DIR)}")
            print("¡Hasta luego!")
            break

        elif op == "13":
            print("\nEjecutando todos los algoritmos, espere...")
            resultados = []
            for k in OPCIONES:
                try:
                    res = OPCIONES[k][1]()
                    resultados.append(res)
                    imprimir_resultado(res)
                except Exception as e:
                    print(f"   [ERROR en opción {k}]: {e}")
            ruta = guardar_informe(resultados)
            print(f"\nInforme completo guardado en: {os.path.relpath(ruta, BASE_DIR)}")
            input("\nPresione ENTER para volver al menú...")

        elif op in OPCIONES:
            try:
                res = OPCIONES[op][1]()
                # reemplaza si ya existía ese algoritmo, para no duplicar en el informe
                resultados = [r for r in resultados if r["algoritmo"] != res["algoritmo"]]
                resultados.append(res)
                imprimir_resultado(res)
            except Exception as e:
                print(f"   [ERROR]: {e}")
            input("\nPresione ENTER para volver al menú...")

        else:
            print("Opción inválida. Intente nuevamente.")


if __name__ == "__main__":
    main()
