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

Segunda parte: ELEVACIONES VS DEPRESIONES. ¿Se comportan distinto los dos signos?
    Características por episodio (salen de las anotaciones de la base):
        duracion_s        fin - inicio. Se excluyen los episodios con fin inferido
                          (seguían abiertos al terminar el registro, así que su
                          duración real es desconocida).
        amplitud_max_mv   valor absoluto de la desviación máxima anotada (mV).
        t_max_s           tiempo al máximo: pico - inicio.
        t_max_rel         t_max_s / duracion_s (0 = máximo al comienzo, 1 = al final).
        t_recuperacion_s  tiempo de recuperación: fin - pico (también sin fin inferido).
    Comparación con Mann-Whitney (dos grupos independientes, sin supuesto de
    normalidad: las duraciones tienen una cola muy larga). Tamaño del efecto: r de
    Cliff = 2U/(n1*n2) - 1 (-1 a 1; positivo si las elevaciones tienden a ser
    mayores que las depresiones). Holm entre las características.
    También se describe el reparto de duración y amplitud en rangos y se comparan
    entre signos los CAMBIOS de HRV (durante - antes y después - antes): en
    diferencia para FC y RMSSD y en log del cociente para HF y LF/HF, que son muy
    asimétricas. Se usan los mismos eventos que en la primera parte.

Uso:
    python src/estadistica.py [base]
Salidas:
    results/estadistica_descriptivos_<base>.csv
    results/estadistica_friedman_<base>.csv
    results/estadistica_pares_<base>.csv
    results/estadistica_episodios_caracteristicas_<base>.csv  una fila por episodio
    results/estadistica_episodios_comparacion_<base>.csv      + vs - por característica
    results/estadistica_episodios_rangos_<base>.csv           % de episodios por rango y signo
    results/estadistica_hrv_signos_<base>.csv                 cambios de HRV: + vs -
    results/figuras/estadistica_episodios_distribuciones_<base>.png
    results/figuras/estadistica_hrv_signos_<base>.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata, wilcoxon

from ensamble import COLORES_SIGNO
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


CARACTERISTICAS = {
    "duracion_s": "Duración (s)",
    "amplitud_max_mv": "Amplitud máxima (mV)",
    "t_max_s": "Tiempo al máximo (s)",
    "t_max_rel": "Tiempo al máximo / duración",
    "t_recuperacion_s": "Tiempo de recuperación (s)",
}


def caracteristicas_episodios(gt_df):
    """Características de cada episodio ST anotado (sin cambios de eje, que son
    cambios de posición y no isquemia).

    Entrada: tabla de episodios (episodios_<base>.csv).
    Salida: DataFrame con registro, derivacion, signo, inicio_s, fin_s y las
    columnas de CARACTERISTICAS. Queda NaN lo que no se puede medir: sin pico
    anotado no hay amplitud ni tiempos, y con fin inferido no hay duración ni
    tiempo de recuperación.
    """
    ep = gt_df[(gt_df["tipo"] == "ST") & ~gt_df["cambio_eje"]].copy()
    fin_valido = ~ep["fin_inferido"]
    sin_pico = ep["pico_s"].isna()

    ep["duracion_s"] = (ep["fin_s"] - ep["inicio_s"]).where(fin_valido)
    ep["amplitud_max_mv"] = ep["desviacion_max_mv"].abs()
    ep["t_max_s"] = ep["pico_s"] - ep["inicio_s"]
    ep["t_max_rel"] = ep["t_max_s"] / ep["duracion_s"]
    ep["t_recuperacion_s"] = (ep["fin_s"] - ep["pico_s"]).where(fin_valido)
    ep.loc[sin_pico, ["amplitud_max_mv", "t_max_s", "t_max_rel", "t_recuperacion_s"]] = np.nan

    columnas = ["registro", "derivacion", "signo", "inicio_s", "fin_s"] + list(CARACTERISTICAS)
    return ep[columnas].reset_index(drop=True)


def comparar_por_signo(tabla, columnas, p):
    """Compara elevaciones (+) y depresiones (-) en cada columna.

    Entradas: tabla con la columna `signo`, columnas (dict columna -> nombre) y
    p (parámetros: n_min). Salida: DataFrame con una fila por característica:
    n, media, desvío, mediana y q25-q75 de cada signo, U de Mann-Whitney, p,
    p (Holm), r de Cliff y la razón de medianas (+ / -). Sin prueba si algún
    signo tiene menos de n_min episodios con dato.
    """
    filas = []
    for col, nombre in columnas.items():
        mas = tabla.loc[tabla["signo"] == "+", col].dropna()
        menos = tabla.loc[tabla["signo"] == "-", col].dropna()
        fila = {"Característica": nombre}
        for s, x in (("+", mas), ("-", menos)):
            q25, med, q75 = x.quantile([0.25, 0.5, 0.75]) if len(x) else (np.nan,) * 3
            fila.update({f"n ({s})": len(x), f"Media ({s})": x.mean(), f"Desvío ({s})": x.std(),
                         f"Mediana ({s})": med, f"q25 ({s})": q25, f"q75 ({s})": q75})
        fila.update({"U": np.nan, "p": np.nan, "r de Cliff": np.nan, "Mediana (+) / (-)": np.nan})
        if len(mas) >= p["n_min"] and len(menos) >= p["n_min"]:
            u, pv = mannwhitneyu(mas, menos, alternative="two-sided")
            fila.update({"U": u, "p": pv, "r de Cliff": 2 * u / (len(mas) * len(menos)) - 1,
                         "Mediana (+) / (-)": fila["Mediana (+)"] / fila["Mediana (-)"]})
        filas.append(fila)
    resultado = pd.DataFrame(filas)
    resultado["p (Holm)"] = holm(resultado["p"])
    return resultado


def distribucion_por_rangos(tabla, columna, bordes, etiquetas):
    """Porcentaje de episodios de cada signo en cada rango de `columna`.

    Entradas: tabla (de caracteristicas_episodios), columna, bordes (cortes; los
    rangos son (a, b], con el primero también cerrado a la izquierda) y
    etiquetas (una por rango). Salida: DataFrame con una fila por rango y las
    columnas "% (+)", "% (-)", "n (+)" y "n (-)". Solo cuentan los episodios con
    dato en `columna`.
    """
    filas = {}
    for s in ("+", "-"):
        x = tabla.loc[tabla["signo"] == s, columna].dropna()
        rango = pd.cut(x, bins=bordes, labels=etiquetas, include_lowest=True)
        n = rango.value_counts().reindex(etiquetas).fillna(0).astype(int)
        filas[f"n ({s})"] = n
        filas[f"% ({s})"] = 100 * n / max(len(x), 1)
    salida = pd.DataFrame(filas)
    salida.insert(0, "Rango", etiquetas)
    salida.insert(0, "Característica", CARACTERISTICAS[columna])
    return salida.reset_index(drop=True)


def rangos_duracion_amplitud(tabla):
    """Reparto de duración y amplitud máxima en rangos, por signo."""
    duracion = distribucion_por_rangos(
        tabla, "duracion_s", [0, 60, 300, 600, np.inf],
        ["≤ 1 min", "1 a 5 min", "5 a 10 min", "> 10 min"])
    amplitud = distribucion_por_rangos(
        tabla, "amplitud_max_mv", [0, 0.1, 0.2, 0.3, np.inf],
        ["≤ 0,1 mV", "0,1 a 0,2 mV", "0,2 a 0,3 mV", "> 0,3 mV"])
    return pd.concat([duracion, amplitud], ignore_index=True)


def figura_distribuciones(tabla, ruta):
    """Una figura con un panel por característica (duración, amplitud, tiempo
    al máximo y tiempo de recuperación): caja (mediana e IQR) y los episodios
    individuales, para elevación y depresión. Los tiempos van en escala
    logarítmica."""
    paneles = [("duracion_s", True), ("amplitud_max_mv", False),
               ("t_max_s", True), ("t_recuperacion_s", True)]
    rng = np.random.default_rng(0)               # dispersión horizontal reproducible
    fig, ejes = plt.subplots(1, len(paneles), figsize=(14, 4.2))
    for ax, (col, logaritmica) in zip(ejes, paneles):
        datos = [tabla.loc[tabla["signo"] == s, col].dropna().to_numpy() for s in ("+", "-")]
        caja = ax.boxplot(datos, positions=[0, 1], widths=0.5, showfliers=False,
                          patch_artist=True, medianprops={"color": "k", "lw": 1.5})
        for parche, s in zip(caja["boxes"], ("+", "-")):
            parche.set(facecolor=COLORES_SIGNO[s], alpha=0.25, edgecolor=COLORES_SIGNO[s])
        for i, (x, s) in enumerate(zip(datos, ("+", "-"))):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(x)), x, s=7, color=COLORES_SIGNO[s],
                       alpha=0.45, linewidths=0)
        if logaritmica:
            ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Elevación\n(n = {len(datos[0])})", f"Depresión\n(n = {len(datos[1])})"])
        ax.set_title(CARACTERISTICAS[col])
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def cambios_hrv(tabla_hrv, metricas):
    """Cambios de HRV por episodio, para comparar entre signos.

    Entrada: tabla hrv_<base>.csv y lista de métricas. Salida: (tabla larga con
    una fila por evento y métrica-cambio, dict columna -> nombre). El cambio es
    la diferencia (FC, RMSSD) o el logaritmo del cociente (potencias, LF/HF) de
    durante y de después respecto de antes; se usan solo eventos con las tres
    ventanas válidas en esa métrica (datos_completos).
    """
    partes, nombres = [], {}
    for m in metricas:
        en_log = m not in ("fc_media_lpm", "rmssd_ms")
        for c in ("durante", "despues"):
            col = f"{m}__{c}_vs_antes"
            nombres[col] = f"{NOMBRES_METRICA[m]}: {c} vs antes ({'log del cociente' if en_log else 'diferencia'})"
            filas = []
            for s in ("+", "-"):
                d = datos_completos(tabla_hrv, s, m)
                if en_log:
                    d = d[(d[c] > 0) & (d["antes"] > 0)]
                    valor = np.log(d[c] / d["antes"])
                else:
                    valor = d[c] - d["antes"]
                filas.append(pd.DataFrame({"signo": s, col: valor.to_numpy()}))
            partes.append(pd.concat(filas, ignore_index=True))
    # cada métrica tiene su propio conjunto de eventos: se comparan por separado
    return partes, nombres


def comparar_hrv(tabla_hrv, p):
    """Compara entre signos los cambios de HRV (ver cambios_hrv). Salida:
    DataFrame de comparar_por_signo con p (Holm) entre todas las comparaciones."""
    partes, nombres = cambios_hrv(tabla_hrv, p["metricas"])
    filas = [comparar_por_signo(parte, {col: nombres[col]}, p)
             for parte, col in zip(partes, nombres)]
    resultado = pd.concat(filas, ignore_index=True)
    resultado["p (Holm)"] = holm(resultado["p"])
    return resultado.rename(columns={"Característica": "Cambio"})


def figura_hrv(tabla_hrv, p, ruta):
    """Un panel por métrica: mediana e IQR de antes, durante y después, para
    elevación y depresión (eventos representantes con las tres ventanas
    válidas en la métrica)."""
    metricas = p["metricas"]
    fig, ejes = plt.subplots(1, len(metricas), figsize=(3.6 * len(metricas), 3.8))
    for ax, m in zip(np.atleast_1d(ejes), metricas):
        for k, s in enumerate(("+", "-")):
            d = datos_completos(tabla_hrv, s, m)
            q = d[CONDICIONES].quantile([0.25, 0.5, 0.75])
            x = np.arange(len(CONDICIONES)) + (k - 0.5) * 0.12
            ax.errorbar(x, q.loc[0.5], yerr=[q.loc[0.5] - q.loc[0.25], q.loc[0.75] - q.loc[0.5]],
                        color=COLORES_SIGNO[s], marker="o", capsize=3, lw=1.5,
                        label=f"{NOMBRES_SIGNO[s]} (n = {len(d)})")
        ax.set_xticks(range(len(CONDICIONES)))
        ax.set_xticklabels(["antes", "durante", "después"])
        ax.set_title(NOMBRES_METRICA[m])
        ax.grid(axis="y", color="#e6e6e6")
        ax.spines[["top", "right"]].set_visible(False)
    handles, etiquetas = np.atleast_1d(ejes)[0].get_legend_handles_labels()
    fig.legend(handles, etiquetas, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


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

    # ---------------- Segunda parte: elevaciones vs depresiones ----------------
    print("\n>>>>>>>>>>>   Paso 10 (2): elevaciones vs depresiones de ST   <<<<<<<<<<<<")
    gt_df = pd.read_csv(RESULTS / f"episodios_{base}.csv")
    ep = caracteristicas_episodios(gt_df)
    ep.to_csv(RESULTS / f"estadistica_episodios_caracteristicas_{base}.csv", index=False,
              float_format="%.6g")

    conteo = pd.DataFrame({
        "Episodios": ep.groupby("signo").size(),
        "Registros": ep.groupby("signo")["registro"].nunique(),
        "Fin inferido (sin duración)": ep.groupby("signo")["duracion_s"].apply(lambda s: s.isna().sum()),
    })
    conteo.index = conteo.index.map(NOMBRES_SIGNO)
    print("\n ------------------ Episodios ST anotados (sin cambios de eje) ------------------ ")
    print(conteo.to_string())

    comparacion = comparar_por_signo(ep, CARACTERISTICAS, p)
    comparacion.to_csv(RESULTS / f"estadistica_episodios_comparacion_{base}.csv", index=False,
                       float_format="%.6g")
    print("\n ------------- Elevación (+) vs depresión (-): Mann-Whitney, Holm ------------- ")
    print(comparacion.to_string(index=False, float_format="%.3g"))

    rangos = rangos_duracion_amplitud(ep)
    rangos.to_csv(RESULTS / f"estadistica_episodios_rangos_{base}.csv", index=False, float_format="%.6g")
    print("\n ------------- Reparto de duración y amplitud en rangos (% de cada signo) ------------- ")
    print(rangos.to_string(index=False, float_format="%.1f"))

    (RESULTS / "figuras").mkdir(exist_ok=True)
    figura_distribuciones(ep, RESULTS / "figuras" / f"estadistica_episodios_distribuciones_{base}.png")
    print(f"\nFIGURA estadistica_episodios_distribuciones_{base}.png: duración, amplitud máxima, tiempo "
          "al máximo y tiempo de recuperación de cada episodio, por signo (escala log en los tiempos).")

    hrv = comparar_hrv(tabla, p)
    hrv.to_csv(RESULTS / f"estadistica_hrv_signos_{base}.csv", index=False, float_format="%.6g")
    print("\n ------- Cambios de HRV: elevación (+) vs depresión (-), Mann-Whitney, Holm ------- ")
    print(hrv[["Cambio", "n (+)", "n (-)", "Mediana (+)", "Mediana (-)", "p", "p (Holm)", "r de Cliff"]]
          .to_string(index=False, float_format="%.3g"))
    figura_hrv(tabla, p, RESULTS / "figuras" / f"estadistica_hrv_signos_{base}.png")
    print(f"\nFIGURA estadistica_hrv_signos_{base}.png: mediana e IQR de cada métrica de HRV antes, "
          "durante y después, por signo.")
    print("\nLos episodios de un mismo registro no son independientes: leer los p como "
          "descriptivos y apoyarse en el tamaño del efecto (r de Cliff).")