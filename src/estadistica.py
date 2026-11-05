""" ||||||||||||  10MO PASO: ESTADÍSTICA||||||||||||

Estadística de la HRV alrededor de los episodios ST.

Pregunta: ¿cambia la HRV entre antes, durante y después de un episodio ST?

Diseño: medidas repetidas (las tres condiciones salen del mismo episodio), por
separado para elevación y depresión. Se usan los eventos representantes con las
tres ventanas válidas en la métrica analizada (ver hrv.py).
    1. Friedman por métrica (tres condiciones). Tamaño del efecto: W de Kendall
       (0 a 1) = chi2 / (n * (k - 1)).
    2. Si corresponde, comparaciones de a pares con Wilcoxon de rangos con signo.
       Tamaño del efecto: correlación biserial por rangos (-1 a 1; positiva si la
       segunda condición es mayor que la primera) y mediana de las diferencias.
    3. Corrección de Holm: los p de Friedman se corrigen entre las métricas de
       cada signo, y los de Wilcoxon entre todas las comparaciones de cada signo.

Advertencia (no independencia): los eventos de un mismo registro provienen del
mismo paciente y no son independientes entre sí; además los p de Wilcoxon
dependen de n. Por eso se informan los tamaños del efecto y cuántos registros
distintos aportan los datos, y los p se leen como descriptivos, no como
inferencia sobre una población de pacientes.

Uso:
    python src/estadistica.py [base]
Salidas:
    results/estadistica_descriptivos_<base>.csv
    results/estadistica_friedman_<base>.csv
    results/estadistica_pares_<base>.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from rutas import CONFIG, RESULTS

CONDICIONES = ["antes", "durante", "despues"]
NOMBRES_SIGNO = {"+": "Elevación de ST", "-": "Depresión de ST"}
NOMBRES_METRICA = {"fc_media_lpm": "FC media (lpm)", "rmssd_ms": "RMSSD (ms)",
                   "lf_ms2": "LF (ms²)", "hf_ms2": "HF (ms²)", "lf_hf": "LF/HF",
                   "lf_nu": "LF (u.n.)"}


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/estadistica.json."""
    with open(CONFIG / "estadistica.json", encoding="utf-8") as f:
        return json.load(f)


def holm(p_valores):
    """Corrección de Holm-Bonferroni. Entrada: lista de p (puede tener NaN).
    Salida: array de p ajustados, en el mismo orden (NaN se mantiene)."""
    p = np.asarray(p_valores, dtype=float)
    ajustado = np.full(len(p), np.nan)
    validos = np.where(~np.isnan(p))[0]
    orden = validos[np.argsort(p[validos])]
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, (len(orden) - rango) * p[i])
        ajustado[i] = min(maximo, 1.0)
    return ajustado


def datos_completos(tabla, signo, metrica):
    """Eventos representantes de un signo con las tres ventanas válidas en la
    métrica. Salida: DataFrame con una columna por condición."""
    sub = tabla[(tabla["signo"] == signo) & tabla["representante"]]
    columnas = {metrica + "_" + c: c for c in CONDICIONES}
    return sub.dropna(subset=list(columnas)).rename(columns=columnas)


def biserial_rangos(diferencias):
    """Correlación biserial por rangos de las diferencias (segunda - primera):
    (suma de rangos positivos - suma de rangos negativos) / suma de rangos.
    Las diferencias nulas se descartan."""
    d = diferencias[diferencias != 0]
    if len(d) == 0:
        return np.nan
    rangos = rankdata(np.abs(d))
    return (rangos[d > 0].sum() - rangos[d < 0].sum()) / rangos.sum()


def resumir(tabla, p):
    """Calcula descriptivos, Friedman y comparaciones de a pares.

    Salidas: (descriptivos, friedman, pares), tres DataFrames. Las filas con
    menos de n_min eventos completos quedan sin prueba."""
    desc, fried, pares = [], [], []
    for signo in ("+", "-"):
        for metrica in p["metricas"]:
            d = datos_completos(tabla, signo, metrica)
            base = {"Signo": NOMBRES_SIGNO[signo], "Métrica": NOMBRES_METRICA[metrica],
                    "Eventos": len(d), "Registros": d["registro"].nunique()}
            fila = {**base}
            for c in CONDICIONES:
                if len(d):
                    q25, med, q75 = d[c].quantile([0.25, 0.5, 0.75])
                    fila["Mediana " + c] = med
                    fila["IQR " + c] = q75 - q25
                else:
                    fila["Mediana " + c], fila["IQR " + c] = np.nan, np.nan
            desc.append(fila)

            f = {**base, "chi2": np.nan, "p": np.nan, "W de Kendall": np.nan}
            if len(d) >= p["n_min"]:
                chi2, pv = friedmanchisquare(*[d[c] for c in CONDICIONES])
                f.update({"chi2": chi2, "p": pv, "W de Kendall": chi2 / (len(d) * (len(CONDICIONES) - 1))})
            fried.append(f)

            for a, b in p["comparaciones"]:
                par = {**base, "Comparación": b + " - " + a, "W": np.nan, "p": np.nan,
                       "Biserial por rangos": np.nan, "Mediana de la diferencia": np.nan}
                if len(d) >= p["n_min"]:
                    dif = (d[b] - d[a]).to_numpy()
                    par["Mediana de la diferencia"] = float(np.median(dif))
                    par["Biserial por rangos"] = biserial_rangos(dif)
                    if np.any(dif != 0):
                        par["W"], par["p"] = wilcoxon(d[b], d[a])
                pares.append(par)

    desc, fried, pares = pd.DataFrame(desc), pd.DataFrame(fried), pd.DataFrame(pares)
    for signo in NOMBRES_SIGNO.values():
        m = fried["Signo"] == signo
        fried.loc[m, "p (Holm)"] = holm(fried.loc[m, "p"])
        m = pares["Signo"] == signo
        pares.loc[m, "p (Holm)"] = holm(pares.loc[m, "p"])
    return desc, fried, pares


if __name__ == "__main__":
    print("\n>>>>>>>>>>>   Paso 10: estadística de la HRV   <<<<<<<<<<<<")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    p = leer_parametros()
    tabla = pd.read_csv(RESULTS / f"hrv_{base}.csv")
    desc, fried, pares = resumir(tabla, p)

    desc.to_csv(RESULTS / f"estadistica_descriptivos_{base}.csv", index=False, float_format="%.6g")
    fried.to_csv(RESULTS / f"estadistica_friedman_{base}.csv", index=False, float_format="%.6g")
    pares.to_csv(RESULTS / f"estadistica_pares_{base}.csv", index=False, float_format="%.6g")

    print("\n ========================== Descriptivos: mediana e IQR por condición ==========================")
    print(desc.to_string(index=False, float_format="%.2f"))
    print("\n ==================== Friedman (antes, durante, después) ====================")
    print(fried.to_string(index=False, float_format="%.3f"))
    print("\n ========================== Comparaciones de a pares (Wilcoxon, Holm) ==========================")
    print(pares.to_string(index=False, float_format="%.3f"))
    print("\nLos eventos de un mismo registro no son independientes: leer los p como "
          "descriptivos y apoyarse en los tamaños del efecto.")