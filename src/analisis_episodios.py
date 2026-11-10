""" ||||||||||||  11VO PASO: ELEVACIONES VS DEPRESIONES DE ST ||||||||||||

Comparación de las características de los episodios de elevación (+) y de
depresión (-) del segmento ST.

Pregunta: ¿se comportan distinto las elevaciones y las depresiones?

Características por episodio (salen de las anotaciones de la base):
    duracion_s        fin - inicio. Se excluyen los episodios con fin inferido
                      (el episodio seguía abierto al terminar el registro, así
                      que su duración real es desconocida).
    amplitud_max_mv   valor absoluto de la desviación máxima anotada (mV).
    t_max_s           tiempo que tarda en llegar al máximo: pico - inicio.
    t_max_rel         t_max_s / duracion_s (en qué fracción del episodio ocurre
                      el máximo; 0 = al comienzo, 1 = al final).
    t_recuperacion_s  tiempo de recuperación: fin - pico. Se excluyen los de fin
                      inferido por la misma razón que la duración.

Comparación: Mann-Whitney (dos grupos independientes, sin supuesto de
normalidad: las duraciones tienen una cola muy larga). Tamaño del efecto: r de
Cliff / biserial por rangos = 2U/(n1*n2) - 1 (-1 a 1; positivo si las elevaciones
tienden a ser mayores que las depresiones). Corrección de Holm entre las
características.

Advertencia (no independencia): un mismo evento puede estar anotado en las dos
derivaciones y un registro aporta varios episodios. Los p se leen como
descriptivos, no como inferencia sobre una población de pacientes (igual que en
estadistica.py).

Además se describe cómo se reparten las duraciones y las amplitudes en rangos
(qué proporción de cada signo cae en cada rango) y se dibujan las
distribuciones. Las duraciones y los tiempos tienen una cola larga (unos pocos
episodios de más de una hora), por eso la figura usa escala logarítmica.

Perfil temporal del ST (ensamble_<base>.csv, ya calculado por signo): se
describe la curva mediana de cada signo con la amplitud y el instante del pico,
el instante en que alcanza fraccion_latencia del pico (puede ser anterior al
inicio anotado) y el instante en que vuelve por debajo de esa fracción. La curva normalizada por su pico permite
comparar la forma sin que influya la amplitud. Ojo: el ensamble se alinea al
inicio y los episodios duran distinto, así que después del pico la mediana
mezcla episodios que ya terminaron con otros que siguen.

HRV antes, durante y después (hrv_<base>.csv): se usan los mismos eventos que
en estadistica.py (representantes con las tres ventanas válidas en la métrica).
Se comparan entre signos los CAMBIOS (durante - antes y después - antes): en
diferencia para FC y RMSSD y en cociente logarítmico para las potencias y LF/HF,
que son muy asimétricas. Mann-Whitney y Holm entre todas las comparaciones.

Uso:
    python src/analisis_episodios.py [base]
Salidas:
    results/analisis_episodios_caracteristicas_<base>.csv   una fila por episodio
    results/analisis_episodios_comparacion_<base>.csv       comparación por signo
    results/analisis_episodios_rangos_<base>.csv            % de episodios por rango y signo
    results/analisis_episodios_perfil_st_<base>.csv         descriptores del ensamble por signo
    results/analisis_episodios_hrv_<base>.csv               cambios de HRV: + vs -
    results/figuras/analisis_episodios_distribuciones_<base>.png
    results/figuras/analisis_episodios_perfil_st_<base>.png
    results/figuras/analisis_episodios_hrv_<base>.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ensamble import COLORES_SIGNO
from estadistica import (CONDICIONES, NOMBRES_METRICA, NOMBRES_SIGNO, datos_completos, holm,
                         leer_parametros as leer_parametros_estadistica)
from rutas import CONFIG, RESULTS

CARACTERISTICAS = {
    "duracion_s": "Duración (s)",
    "amplitud_max_mv": "Amplitud máxima (mV)",
    "t_max_s": "Tiempo al máximo (s)",
    "t_max_rel": "Tiempo al máximo / duración",
    "t_recuperacion_s": "Tiempo de recuperación (s)",
}


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/analisis_episodios.json."""
    with open(CONFIG / "analisis_episodios.json", encoding="utf-8") as f:
        return json.load(f)


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


def descriptores_perfil(ensamble, p):
    """Descriptores de la curva mediana del ensamble de cada signo.

    Entradas: tabla de ensamble_<base>.csv y p (fraccion_latencia). Salida:
    DataFrame con una fila por signo: amplitud del pico (mV, con signo), su
    instante (s; el pico se busca desde t = 0), n de episodios con dato en el
    pico, instante en que |mediana| alcanza por primera vez la fracción del pico
    (puede ser negativo: la anotación marca el cruce del umbral, y el ST ya
    venía moviéndose), tiempo de subida (pico - ese instante), instante de
    retorno (primer instante posterior al pico con |mediana| por debajo de la
    fracción; NaN si no vuelve en la ventana) y tiempo de retorno desde el pico.
    """
    filas = []
    for s in ("+", "-"):
        e = ensamble[ensamble["signo"] == s].dropna(subset=["st_mediana_mv"]).reset_index(drop=True)
        desde_cero = e[e["t_relativo_s"] >= 0]
        if desde_cero.empty:
            continue
        magnitud = e["st_mediana_mv"].abs()
        i_pico = int(magnitud[desde_cero.index].idxmax())
        umbral = p["fraccion_latencia"] * magnitud[i_pico]
        t_pico = e.loc[i_pico, "t_relativo_s"]
        t_sube = e.loc[e.index[(magnitud >= umbral) & (e.index <= i_pico)][0], "t_relativo_s"]
        despues = e.index[(e.index > i_pico) & (magnitud < umbral)]
        t_retorno = e.loc[despues[0], "t_relativo_s"] if len(despues) else np.nan
        filas.append({"Signo": NOMBRES_SIGNO[s], "Amplitud del pico (mV)": e.loc[i_pico, "st_mediana_mv"],
                      "Instante del pico (s)": t_pico, "Episodios en el pico": int(e.loc[i_pico, "n_episodios"]),
                      "Instante en que alcanza la fracción del pico (s)": t_sube,
                      "Tiempo de subida hasta el pico (s)": t_pico - t_sube,
                      "Instante de retorno (s)": t_retorno,
                      "Tiempo de retorno desde el pico (s)": t_retorno - t_pico})
    return pd.DataFrame(filas)


def figura_perfil(ensamble, ruta):
    """Dos paneles: mediana e IQR del cambio de ST de elevaciones y depresiones
    superpuestas (izquierda) y la mediana normalizada por el valor absoluto de su
    pico, para comparar la forma (derecha)."""
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(12, 4.2))
    for s in ("+", "-"):
        e = ensamble[ensamble["signo"] == s]
        color = COLORES_SIGNO[s]
        ax.plot(e["t_relativo_s"], e["st_mediana_mv"], color=color, lw=2, label=NOMBRES_SIGNO[s])
        ax.fill_between(e["t_relativo_s"], e["st_p25_mv"], e["st_p75_mv"], color=color, alpha=0.2)
        pico = e.loc[e["t_relativo_s"] >= 0, "st_mediana_mv"].abs().max()
        axn.plot(e["t_relativo_s"], e["st_mediana_mv"].abs() / pico, color=color, lw=2,
                 label=NOMBRES_SIGNO[s])
    for a in (ax, axn):
        a.axvline(0, color="k", ls="--", alpha=0.7)
        a.set_xlabel("Tiempo relativo al inicio (s)")
        a.grid(color="#e6e6e6")
        a.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Cambio de ST (mV): mediana e IQR")
    ax.legend(frameon=False)
    axn.set_ylabel("|mediana| / |pico|")
    axn.set_title("Forma del perfil (normalizado por su pico)")
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


def comparar_hrv(tabla_hrv, p_est, p):
    """Compara entre signos los cambios de HRV (ver cambios_hrv). Salida:
    DataFrame de comparar_por_signo con p (Holm) entre todas las comparaciones."""
    partes, nombres = cambios_hrv(tabla_hrv, p_est["metricas"])
    filas = [comparar_por_signo(parte, {col: nombres[col]}, p)
             for parte, col in zip(partes, nombres)]
    resultado = pd.concat(filas, ignore_index=True)
    resultado["p (Holm)"] = holm(resultado["p"])
    return resultado.rename(columns={"Característica": "Cambio"})


def figura_hrv(tabla_hrv, p_est, ruta):
    """Un panel por métrica: mediana e IQR de antes, durante y después, para
    elevación y depresión (eventos representantes con las tres ventanas
    válidas en la métrica)."""
    metricas = p_est["metricas"]
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
    print("\n>>>>>>>>>>>   Paso 11: elevaciones vs depresiones de ST   <<<<<<<<<<<<")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    p = leer_parametros()
    gt_df = pd.read_csv(RESULTS / f"episodios_{base}.csv")

    tabla = caracteristicas_episodios(gt_df)
    tabla.to_csv(RESULTS / f"analisis_episodios_caracteristicas_{base}.csv", index=False,
                 float_format="%.6g")

    conteo = pd.DataFrame({
        "Episodios": tabla.groupby("signo").size(),
        "Registros": tabla.groupby("signo")["registro"].nunique(),
        "Fin inferido (sin duración)": tabla.groupby("signo")["duracion_s"].apply(lambda s: s.isna().sum()),
    })
    conteo.index = conteo.index.map(NOMBRES_SIGNO)
    print("\n ------------------ Episodios ST anotados (sin cambios de eje) ------------------ ")
    print(conteo.to_string())

    comparacion = comparar_por_signo(tabla, CARACTERISTICAS, p)
    comparacion.to_csv(RESULTS / f"analisis_episodios_comparacion_{base}.csv", index=False,
                       float_format="%.6g")
    print("\n ------------- Elevación (+) vs depresión (-): Mann-Whitney, Holm ------------- ")
    print(comparacion.to_string(index=False, float_format="%.3g"))
    print("\nLos episodios de un mismo registro no son independientes: leer los p como "
          "descriptivos y apoyarse en el tamaño del efecto (r de Cliff).")

    rangos = rangos_duracion_amplitud(tabla)
    rangos.to_csv(RESULTS / f"analisis_episodios_rangos_{base}.csv", index=False, float_format="%.6g")
    print("\n ------------- Reparto de duración y amplitud en rangos (% de cada signo) ------------- ")
    print(rangos.to_string(index=False, float_format="%.1f"))

    (RESULTS / "figuras").mkdir(exist_ok=True)
    figura_distribuciones(tabla, RESULTS / "figuras" / f"analisis_episodios_distribuciones_{base}.png")
    print(f"\nFIGURA analisis_episodios_distribuciones_{base}.png: duración, amplitud máxima, tiempo "
          "al máximo y tiempo de recuperación de cada episodio, por signo (escala log en los tiempos).")

    # Perfil temporal del ST (ensamble por signo).
    ensamble = pd.read_csv(RESULTS / f"ensamble_{base}.csv")
    perfil = descriptores_perfil(ensamble, p)
    perfil.to_csv(RESULTS / f"analisis_episodios_perfil_st_{base}.csv", index=False, float_format="%.6g")
    print("\n ------------- Perfil temporal del ST: descriptores de la curva mediana ------------- ")
    print(perfil.to_string(index=False, float_format="%.3g"))
    figura_perfil(ensamble, RESULTS / "figuras" / f"analisis_episodios_perfil_st_{base}.png")
    print(f"\nFIGURA analisis_episodios_perfil_st_{base}.png: perfil mediano del ST de elevaciones y "
          "depresiones, en mV y normalizado por su pico.")

    # HRV antes, durante y después.
    p_est = leer_parametros_estadistica()
    tabla_hrv = pd.read_csv(RESULTS / f"hrv_{base}.csv")
    hrv = comparar_hrv(tabla_hrv, p_est, p)
    hrv.to_csv(RESULTS / f"analisis_episodios_hrv_{base}.csv", index=False, float_format="%.6g")
    print("\n ------- Cambios de HRV: elevación (+) vs depresión (-), Mann-Whitney, Holm ------- ")
    print(hrv[["Cambio", "n (+)", "n (-)", "Mediana (+)", "Mediana (-)", "p", "p (Holm)", "r de Cliff"]]
          .to_string(index=False, float_format="%.3g"))
    figura_hrv(tabla_hrv, p_est, RESULTS / "figuras" / f"analisis_episodios_hrv_{base}.png")
    print(f"\nFIGURA analisis_episodios_hrv_{base}.png: mediana e IQR de cada métrica de HRV antes, "
          "durante y después, por signo.")
    print(f"\nResultados guardados en results/analisis_episodios_*_{base}.csv")
