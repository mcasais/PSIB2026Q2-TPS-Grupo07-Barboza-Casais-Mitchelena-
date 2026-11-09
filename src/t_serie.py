""" |||||||||||| 6TO PASO (B): SERIE DE LA ONDA T ||||||||||||

Serie temporal de la amplitud de la onda T latido a latido e interpolada.
Usa la misma cadena de filtrado, los mismos latidos normales, el mismo
isoeléctrico (tramo plano del PR) y el mismo suavizado que la serie ST.
Se usa para analizar la evolución de la onda T en el tiempo, por ejemplo para detectar isquemia aguda. La serie ST se analiza para buscar corrimientos, la serie T para ver si la onda T se invierte o se hace más alta, es decir, si se deforma.

Uso:
    python src/t_serie.py [base]
Salida:
    results/t_serie_<base>.csv.gz
    results/figuras/t_latidos_alineados_<base>.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from episodios import leer_config
from exploracion import mascara_descarte
from filtrado import aplicar_cadena, nivel_isoelectrico, estilo_eje
from filtrado import leer_parametros as leer_parametros_filtrado
from qrs import leer_senal_mv
from rutas import RESULTS, DATOS
from st_serie import leer_parametros, latidos_normales


def medir_t_latido(x_mv, fs, idx_qrs, p):
    """Mide la amplitud de la onda T de cada latido respecto del tramo isoeléctrico.

    Entradas: x_mv (señal filtrada de una derivación, mV), fs (Hz), idx_qrs
    (muestras de todos los QRS detectados) y p (parámetros de config/st.json).
    Para cada latido: la referencia es la misma que en el ST (nivel_isoelectrico);
    se promedia la señal en ventanas de ancho_ms dentro de [R + ventana_ms[0],
    R + ventana_ms[1]] y se toma el extremo de mayor valor absoluto, conservando
    el signo (T positiva o invertida), menos la referencia.
    Salida: DataFrame con t_s (s) y t_mv (mV, con signo) de los latidos cuya
    ventana entra en la señal.
    """
    _, iso_mv, validos = nivel_isoelectrico(x_mv, fs, idx_qrs, p)
    idx = idx_qrs[validos]
    ini = int(round(p["onda_t"]["ventana_ms"][0] / 1000 * fs))
    fin = int(round(p["onda_t"]["ventana_ms"][1] / 1000 * fs))
    ancho = max(int(round(p["onda_t"]["ancho_ms"] / 1000 * fs)), 1)

    dentro = idx + fin < len(x_mv)
    idx, iso_mv = idx[dentro], iso_mv[dentro]

    tramos = x_mv[idx[:, None] + np.arange(ini, fin + 1)]
    promediado = uniform_filter1d(tramos, size=ancho, axis=1, mode="nearest")
    promediado = promediado - iso_mv[:, None]
    extremo = np.argmax(np.abs(promediado), axis=1)
    amplitud = promediado[np.arange(len(idx)), extremo]
    return pd.DataFrame({"t_s": idx / fs, "t_mv": amplitud})


def t_por_latido(x_k, fs, idx_qrs, normales, descartada, p, p_filt):
    """Amplitud de T latido a latido de una derivación.

    Filtra con la cadena elegida (solo latidos normales), mide con todos los
    latidos detectados y se queda con los normales.
    Salidas: df_t (latidos normales medidos) y df_val (los que además están
    fuera de ventanas descartadas y tienen medición).
    """
    x_filt = aplicar_cadena(x_k, fs, idx_qrs[normales], descartada,
                            p["cadena_filtrado"], p_filt)
    df_t = medir_t_latido(x_filt, fs, idx_qrs, p)
    idx_lat = np.round(df_t["t_s"] * fs).astype(int)
    df_t = df_t[np.isin(idx_lat, idx_qrs[normales])]
    idx_validos = ~descartada[np.round(df_t["t_s"] * fs).astype(int)]
    return df_t, df_t[idx_validos].dropna()


def procesar_registro_t(base, registro, latidos, ventanas, p, p_filt):
    """Serie de amplitud de T uniforme de cada derivación de un registro.

    Mismo procesamiento que st_serie.procesar_registro: saca los latidos no
    normales y los de ventanas descartadas, suaviza con mediana móvil y
    interpola a la grilla de fs_remuestreo_hz. Quedan NaN en las ventanas
    descartadas y donde haya más de max_hueco_s entre latidos. Se omiten las
    derivaciones con menos de 10 latidos detectados, normales o válidos.
    Salidas: DataFrame con registro, derivacion, nombre_derivacion, t_s y t_mv,
    y lista de dicts con el conteo de latidos de cada derivación.
    """
    x_mv, fs, nombres = leer_senal_mv(base, registro)
    grid_t = np.arange(0, len(x_mv) / fs, 1.0 / p["fs_remuestreo_hz"])

    filas, resumen = [], []
    for k, nombre in enumerate(nombres):
        idx_qrs = latidos.loc[(latidos["registro"] == registro)
                              & (latidos["derivacion"] == k), "idx_qrs"].to_numpy()
        fila = {"registro": registro, "derivacion": k, "nombre_derivacion": nombre,
                "detectados": len(idx_qrs), "normales": 0, "usados": 0,
                "estado": "ok"}
        resumen.append(fila)

        if len(idx_qrs) < 10:
            fila["estado"] = "menos de 10 latidos detectados"
            continue
        normales = latidos_normales(base, registro, idx_qrs, fs, p)
        fila["normales"] = int(normales.sum())
        if normales.sum() < 10:
            fila["estado"] = "menos de 10 latidos normales"
            continue

        descartada = mascara_descarte(ventanas, registro, k, len(x_mv), fs)
        df_t, df_val = t_por_latido(x_mv[:, k], fs, idx_qrs, normales,
                                    descartada, p, p_filt)
        fila["usados"] = len(df_val)
        if len(df_val) < 10:
            fila["estado"] = "menos de 10 latidos válidos"
            continue

        t_suave = (df_val["t_mv"]
                   .rolling(p["suavizado_latidos"], center=True, min_periods=1)
                   .median())
        f_interp = interp1d(df_val["t_s"], t_suave, kind="linear",
                            bounds_error=False, fill_value=np.nan)
        t_grid = f_interp(grid_t)

        t_lat = df_val["t_s"].to_numpy()
        pos = np.searchsorted(t_lat, grid_t)
        hay_vecinos = (pos > 0) & (pos < len(t_lat))
        hueco_s = np.full(len(grid_t), np.inf)
        hueco_s[hay_vecinos] = t_lat[pos[hay_vecinos]] - t_lat[pos[hay_vecinos] - 1]
        t_grid[hueco_s > p["max_hueco_s"]] = np.nan
        t_grid[descartada[np.round(grid_t * fs).astype(int)]] = np.nan

        filas.append(pd.DataFrame({
            "registro": registro,
            "derivacion": k,
            "nombre_derivacion": nombre,
            "t_s": grid_t,
            "t_mv": t_grid,
        }))

    tabla = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    return tabla, resumen


def latidos_alineados(x_filt, fs, idx, p, desde_ms, hasta_ms):
    """Latidos recortados alrededor de R (de R + desde_ms a R + hasta_ms), cada
    uno con su isoeléctrico del PR restado. Salida: matriz (latidos, muestras)
    y vector de tiempos en ms respecto de R."""
    _, iso_mv, validos = nivel_isoelectrico(x_filt, fs, idx, p)
    idx = idx[validos]
    ini = int(round(desde_ms / 1000 * fs))
    fin = int(round(hasta_ms / 1000 * fs))
    dentro = (idx + ini >= 0) & (idx + fin < len(x_filt))
    idx, iso_mv = idx[dentro], iso_mv[dentro]
    matriz = x_filt[idx[:, None] + np.arange(ini, fin + 1)] - iso_mv[:, None]
    t_ms = np.arange(ini, fin + 1) / fs * 1000
    return matriz, t_ms


def figura_latidos_alineados(base, episodios, latidos, ventanas, p, p_filt, salida):
    """Latido medio (mediana e IQR) alineado en R de las derivaciones con más
    episodios T anotados: en la referencia (primeros referencia_s segundos) y
    dentro del primer episodio T anotado. La banda violeta es la ventana donde
    medir_t_latido busca el extremo de T (ventana_ms)."""
    eps_t = episodios[(episodios["tipo"] == "T") & ~episodios["cambio_eje"]]
    top = (eps_t.groupby(["registro", "derivacion"]).size()
           .nlargest(p["onda_t"]["n_ejemplos"]))
    n = len(top)
    fig, ejes = plt.subplots((n + 1) // 2, 2, figsize=(11, 3.6 * ((n + 1) // 2)),
                             squeeze=False)
    ejes = ejes.ravel()

    for eje, (registro, derivacion) in zip(ejes, top.index):
        x_mv, fs, nombres = leer_senal_mv(base, registro)
        idx_qrs = latidos.loc[(latidos["registro"] == registro)
                              & (latidos["derivacion"] == derivacion), "idx_qrs"].to_numpy()
        normales = latidos_normales(base, registro, idx_qrs, fs, p)
        descartada = mascara_descarte(ventanas, registro, derivacion, len(x_mv), fs)
        x_filt = aplicar_cadena(x_mv[:, derivacion], fs, idx_qrs[normales], descartada,
                                p["cadena_filtrado"], p_filt)
        idx_n = idx_qrs[normales]
        idx_n = idx_n[~descartada[idx_n]]
        t_lat = idx_n / fs

        ep = eps_t[(eps_t["registro"] == registro)
                   & (eps_t["derivacion"] == derivacion)].iloc[0]
        grupos = [("Referencia (primeros " + str(p["onda_t"]["referencia_s"]) + " s)",
                   t_lat < p["onda_t"]["referencia_s"], "#999999"),
                  ("Episodio T anotado", (t_lat >= ep["inicio_s"]) & (t_lat <= ep["fin_s"]),
                   "#7b4fc9")]
        for nombre_grupo, mascara, color in grupos:
            if mascara.sum() < 5:
                continue
            matriz, t_ms = latidos_alineados(x_filt, fs, idx_n[mascara], p, -200, 500)
            q1, med, q3 = np.percentile(matriz, [25, 50, 75], axis=0)
            eje.fill_between(t_ms, q1, q3, color=color, alpha=0.25, linewidth=0)
            eje.plot(t_ms, med, color=color, linewidth=1.8,
                     label=nombre_grupo + " (n=" + str(len(matriz)) + ")")

        eje.axvspan(p["onda_t"]["ventana_ms"][0], p["onda_t"]["ventana_ms"][1],
                    color="#b9a5e6", alpha=0.2, zorder=0, label="Ventana de medición de T")
        eje.axhline(0, color="black", linewidth=0.5)
        eje.set_title(registro + ", derivación " + nombres[derivacion], loc="left")
        eje.set_xlabel("Tiempo respecto de R (ms)")
        eje.set_ylabel("Amplitud (mV)")
        eje.legend(frameon=False, fontsize=7, loc="upper right")
        estilo_eje(eje)

    for eje in ejes[n:]:
        eje.axis("off")
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("\n>>>>>>>>>>>   Paso 6B: serie de la onda T   <<<<<<<<<<<\n")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    cfg = leer_config(base)
    p = leer_parametros()
    p_filt = leer_parametros_filtrado()
    registros = (DATOS / cfg["carpeta"] / cfg["archivo_registros"]).read_text().split()

    latidos = pd.read_csv(RESULTS / f"qrs_{base}.csv.gz")
    ventanas = pd.read_csv(RESULTS / f"calidad_ventanas_{base}.csv.gz")

    res, resumen = [], []
    for r in registros:
        df_r, resumen_r = procesar_registro_t(base, r, latidos, ventanas, p, p_filt)
        resumen.extend(resumen_r)
        if not df_r.empty:
            res.append(df_r)

    df_final = pd.concat(res, ignore_index=True)
    df_final.to_csv(RESULTS / f"t_serie_{base}.csv.gz", index=False, float_format="%.6g")

    resumen = pd.DataFrame(resumen)
    ok = resumen[resumen["estado"] == "ok"]
    detectados = ok["detectados"].sum()
    normales = ok["normales"].sum()
    usados = ok["usados"].sum()
    cantidades = [detectados, detectados - normales, normales - usados, usados]
    tabla_latidos = pd.DataFrame(
        {"Latidos": cantidades,
         "% de detectados": [round(100 * c / detectados, 2) for c in cantidades]},
        index=["Detectados",
               "Excluidos: no normales o sin latido anotado",
               "Excluidos: ventana descartada o sin medición",
               "Usados en la serie T"])
    print("\n ////////// Latidos excluidos de la serie T //////////")
    print(tabla_latidos.to_string())

    carpeta_fig = RESULTS / "figuras"
    carpeta_fig.mkdir(parents=True, exist_ok=True)
    episodios = pd.read_csv(RESULTS / f"episodios_{base}.csv")
    figura_latidos_alineados(base, episodios, latidos, ventanas, p, p_filt,
                             carpeta_fig / f"t_latidos_alineados_{base}.png")
    print("Serie T guardada en t_serie_" + base + ".csv.gz")
    print(">>>>>>>>>>>>>>><<<<<<<<<<<<<<<")