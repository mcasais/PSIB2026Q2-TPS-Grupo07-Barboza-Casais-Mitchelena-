""" |||||||||||| 6TO PASO: SERIE ST ||||||||||||

Serie temporal del nivel de ST latido a latido e interpolada.

Uso:
    python src/st_serie.py [base]
Salida:
    results/st_serie_<base>.csv.gz
"""
import json
import sys
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from episodios import leer_config
from exploracion import mascara_descarte, SIMBOLOS_LATIDO
from filtrado import aplicar_cadena, nivel_isoelectrico, estilo_eje
from filtrado import leer_parametros as leer_parametros_filtrado
from qrs import leer_senal_mv
from rutas import CONFIG, RESULTS, DATOS

def leer_parametros():
    """Devuelve el diccionario de parámetros de config/st.json."""
    with open(CONFIG / 'st.json', encoding='utf-8') as f:
        return json.load(f)

def medir_st_latido(x_mv, fs, idx_qrs, p):
    """Mide el nivel de ST de cada latido respecto del tramo isoeléctrico.

    Entradas: x_mv (señal filtrada de una derivación, mV), fs (Hz), idx_qrs
    (muestras de todos los QRS detectados) y p (parámetros de config/st.json).
    Para cada latido: la referencia es el tramo más plano del PR
    (nivel_isoelectrico); la FC es 60 / media centrada de n_latidos_fc RR; el
    ST es el promedio de una ventana de ancho_ms alrededor de R + offset
    (offset_normal_ms, u offset_taquicardia_ms si la FC supera el umbral)
    menos la referencia.
    Salida: DataFrame con t_s (s), st_mv (mV, NaN si la ventana no entra en
    la señal) y fc_bpm.
    """

    n_samples = len(x_mv)
    
    _, iso_mv, validos = nivel_isoelectrico(x_mv, fs, idx_qrs, p)
    idx = idx_qrs[validos]
    
    #Cálculo de la duración de los intervalos RR y de la frecuencia cardíaca para cada latido. Se usa para decidir si el latido es taquicárdico o no, y así medir el ST en un punto más alejado del QRS.
    rr_s = np.diff(idx) / fs
    rr_s = np.append(rr_s, rr_s[-1] if len(rr_s) > 0 else 0.8)
    # Un RR mayor a rr_max_s no es un latido consecutivo (hay un hueco por
    # ventana descartada o latido no detectado): se ignora en el promedio.
    rr_s[rr_s > p["medicion_st"]["rr_max_s"]] = np.nan
    rr_medio_s = (pd.Series(rr_s)
                  .rolling(p["medicion_st"]["n_latidos_fc"], center=True, min_periods=1)
                  .mean().to_numpy())
    fc_bpm = 60.0 / np.maximum(rr_medio_s, 0.3)
    
    #Después de tener la referencia, se mide el nivel de ST en cada latido, en un punto que depende de si el latido es taquicárdico o no. Se toma la media de la señal en una ventana centrada en ese punto y se resta el nivel isoelectrico.
    st_mv = np.zeros(len(idx))
    ancho_st = int(round(p['medicion_st']['ancho_ms'] / 1000 * fs)) // 2
    
    for i, (qrs_i, fc, iso) in enumerate(zip(idx, fc_bpm, iso_mv)):
        offset_ms = (p['medicion_st']['offset_taquicardia_ms'] 
                     if fc >= p['medicion_st']['umbral_taquicardia_bpm'] 
                     else p['medicion_st']['offset_normal_ms'])
        
        #EXPLICACIÓN DE PORQUE HICIMOS ESTO: Si la FC es => 120bpm (taquicardia), el punto J se acerca a la onda R y el ST se mide a 100ms (J+60ms).De lo contrario, se mide a 120ms (J+80ms). Por eso, el offset depende de la FC. El ancho de la ventana es fijo (20ms).
        #Hacemos esto porque a frecuencias cardíacas altas, el segmento ST se acorta y necesitamos medirlo más cerca del QRS para obtener una medición precisa, sino el punto J (donde termina el QRS) cae en la onda T y no en el segmento.
        centro = int(round(offset_ms / 1000 * fs))
        if qrs_i + centro + ancho_st < n_samples:
            tramo_st = x_mv[qrs_i + centro - ancho_st : qrs_i + centro + ancho_st + 1]
            st_mv[i] = tramo_st.mean() - iso
        else:
            st_mv[i] = np.nan
            
    return pd.DataFrame({'t_s': idx / fs, 'st_mv': st_mv, 'fc_bpm': fc_bpm})

def latidos_normales(base, registro, idx_qrs, fs, p):
    """Marca los QRS detectados que corresponden a un latido anotado como
    normal en la base.

    A cada QRS detectado se le asigna el latido anotado más cercano. Se
    descarta si está a más de `tolerancia_latido_s` (detección sin latido
    real) o si su símbolo no está en `simbolos_validos` (extrasístole,
    escape, marcapaso, no clasificado).
    Salida: vector booleano del largo de idx_qrs.
    """
    cfg = leer_config(base)
    anot = wfdb.rdann(str(DATOS / cfg["carpeta"] / registro),
                      cfg["extension_anotaciones"])
    simbolos = np.array(anot.symbol)
    muestras = np.asarray(anot.sample)
    es_latido = np.isin(simbolos, list(SIMBOLOS_LATIDO))
    simbolos, muestras = simbolos[es_latido], muestras[es_latido]
    if len(muestras) == 0:
        return np.zeros(len(idx_qrs), dtype=bool)

    pos = np.searchsorted(muestras, idx_qrs)
    izq = np.clip(pos - 1, 0, len(muestras) - 1)
    der = np.clip(pos, 0, len(muestras) - 1)
    cerca = np.where(np.abs(muestras[izq] - idx_qrs) <= np.abs(muestras[der] - idx_qrs),
                     izq, der)
    n_tol = int(round(p["tolerancia_latido_s"] * fs))
    return ((np.abs(muestras[cerca] - idx_qrs) <= n_tol)
            & np.isin(simbolos[cerca], list(p["simbolos_validos"])))

def st_por_latido(x_k, fs, idx_qrs, normales, descartada, p, p_filt):
    """ST latido a latido de una derivación.

    Filtra con la cadena elegida (solo latidos normales), mide el ST con todos
    los latidos detectados y se queda con los normales.
    Salidas: df_st (latidos normales medidos) y df_val (los que además están
    fuera de ventanas descartadas y tienen medición).
    """
    x_filt = aplicar_cadena(x_k, fs, idx_qrs[normales], descartada,
                            p["cadena_filtrado"], p_filt)
    df_st = medir_st_latido(x_filt, fs, idx_qrs, p)
    idx_lat = np.round(df_st["t_s"] * fs).astype(int)
    df_st = df_st[np.isin(idx_lat, idx_qrs[normales])]
    idx_validos = ~descartada[np.round(df_st["t_s"] * fs).astype(int)]
    return df_st, df_st[idx_validos].dropna()

def procesar_registro(base, registro, latidos, ventanas, p, p_filt):
    """Serie ST uniforme de cada derivación de un registro.

    Por derivación: filtra con la cadena elegida (usando solo los latidos
    normales), mide el ST latido a latido, saca los latidos no normales y los
    de ventanas descartadas, suaviza con mediana móvil y lo interpola a la
    grilla de fs_remuestreo_hz. Quedan NaN en las ventanas descartadas y donde
    haya más de max_hueco_s entre latidos. Se omiten las derivaciones con
    menos de 10 latidos detectados, normales o válidos.
    Salidas: DataFrame con registro, derivacion, nombre_derivacion, t_s y
    st_mv, y lista de dicts con el conteo de latidos de cada derivación.
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

        # Latidos de ventanas descartadas por calidad de señal.
        descartada = mascara_descarte(ventanas, registro, k, len(x_mv), fs)

        df_st, df_val = st_por_latido(x_mv[:, k], fs, idx_qrs, normales,
                                      descartada, p, p_filt)
        fila["usados"] = len(df_val)
        if len(df_val) < 10:
            fila["estado"] = "menos de 10 latidos válidos"
            continue

        # Mediana móvil de suavizado_latidos latidos: reduce el ruido de la
        # medición latido a latido sin correr los escalones.
        st_suave = (df_val["st_mv"]
                    .rolling(p["suavizado_latidos"], center=True, min_periods=1)
                    .median())
        f_interp = interp1d(df_val["t_s"], st_suave, kind="linear",
                            bounds_error=False, fill_value=np.nan)
        st_grid = f_interp(grid_t)

        # Cada punto de la grilla queda entre dos latidos válidos: si están
        # separados por más de max_hueco_s no hay datos ahí, así que va NaN.
        t_lat = df_val["t_s"].to_numpy()
        pos = np.searchsorted(t_lat, grid_t)
        hay_vecinos = (pos > 0) & (pos < len(t_lat))
        hueco_s = np.full(len(grid_t), np.inf)
        hueco_s[hay_vecinos] = t_lat[pos[hay_vecinos]] - t_lat[pos[hay_vecinos] - 1]
        st_grid[hueco_s > p["max_hueco_s"]] = np.nan
        st_grid[descartada[np.round(grid_t * fs).astype(int)]] = np.nan

        filas.append(pd.DataFrame({
            "registro": registro,
            "derivacion": k,
            "nombre_derivacion": nombre,
            "t_s": grid_t,
            "st_mv": st_grid,
        }))

    tabla = pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
    return tabla, resumen

def figura_serie_st(base, registro, derivacion, serie, latidos, ventanas,
                    episodios, p, p_filt, salida):
    """Ejemplo de la serie ST de una derivación: (a) tacograma, (b) ST de cada
    latido y su suavizado, (c) serie guardada en la grilla uniforme (NaN en los
    huecos). Se centra en el primer episodio ST anotado de esa derivación, o
    en los primeros 20 min si no hay."""
    x_mv, fs, nombres = leer_senal_mv(base, registro)
    idx_qrs = latidos.loc[(latidos["registro"] == registro)
                          & (latidos["derivacion"] == derivacion), "idx_qrs"].to_numpy()
    normales = latidos_normales(base, registro, idx_qrs, fs, p)
    descartada = mascara_descarte(ventanas, registro, derivacion, len(x_mv), fs)
    df_st, df_val = st_por_latido(x_mv[:, derivacion], fs, idx_qrs, normales,
                                  descartada, p, p_filt)
    st_suave = (df_val["st_mv"]
                .rolling(p["suavizado_latidos"], center=True, min_periods=1).median())
    grilla = serie[(serie["registro"] == registro) & (serie["derivacion"] == derivacion)]

    eps = episodios[(episodios["registro"] == registro)
                    & (episodios["derivacion"] == derivacion)
                    & (episodios["tipo"] == "ST") & ~episodios["cambio_eje"]]
    if len(eps) > 0:
        t0 = max(eps["inicio_s"].iloc[0] - 300, 0)
        t1 = min(eps["fin_s"].iloc[0] + 300, t0 + 1800)
    else:
        t0, t1 = 0, 1200

    rr = np.diff(idx_qrs) / fs
    rr[rr > p["medicion_st"]["rr_max_s"]] = np.nan
    t_rr = idx_qrs[1:] / fs
    no_normal = ~normales[1:]

    fig, ejes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ejes[0].plot(t_rr[~no_normal], rr[~no_normal], ".", color="teal",
                 markersize=3, label="Latido normal")
    ejes[0].plot(t_rr[no_normal], rr[no_normal], ".", color="red",
                 markersize=5, label="No normal o sin latido anotado")
    ejes[0].set_ylabel("RR (s)")
    ejes[0].set_title("Tacograma", loc="left")
    ejes[0].legend(frameon=False, fontsize=8, loc="upper right")

    ejes[1].plot(df_val["t_s"], df_val["st_mv"], ".", color="#999999",
                 markersize=3, label="Cada latido")
    ejes[1].plot(df_val["t_s"], st_suave, color="navy", linewidth=1.5,
                 label="Mediana de " + str(p["suavizado_latidos"]) + " latidos")
    ejes[1].set_ylabel("ST (mV)")
    ejes[1].set_title("ST de cada latido (respecto del PR)", loc="left")
    ejes[1].legend(frameon=False, fontsize=8, loc="upper right")

    ejes[2].plot(grilla["t_s"], grilla["st_mv"], color="g", linewidth=1.5)
    ejes[2].set_ylabel("ST (mV)")
    ejes[2].set_xlabel("Tiempo (s)")
    ejes[2].set_title("Serie ST guardada (grilla uniforme de "
                      + str(1 / p["fs_remuestreo_hz"]).replace(".", ",") + " s)", loc="left")

    for i, (_, ep) in enumerate(eps.iterrows()):
        for eje in ejes:
            eje.axvspan(ep["inicio_s"], ep["fin_s"], color="#e6e6e6", zorder=0,
                        label="Episodio ST anotado" if (i == 0 and eje is ejes[2]) else None)
    if len(eps) > 0:
        ejes[2].legend(frameon=False, fontsize=8, loc="upper right")

    ejes[2].set_xlim(t0, t1)
    for eje in ejes:
        estilo_eje(eje)
    fig.suptitle(registro + ", derivación " + nombres[derivacion], fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(salida, dpi=150)
    plt.close(fig)

# Finalmente hacemos todo junto: procesar todos los registros de la base de datos, medir el nivel de ST latido a latido, suavizarlo y remuestrearlo a una frecuencia uniforme. Luego guardamos los resultados en un archivo CSV comprimido.
if __name__ == "__main__":
    print("\n>>>>>>>>>>>   Paso 6: serie ST   <<<<<<<<<<<\n")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    cfg = leer_config(base)
    p = leer_parametros()
    p_filt = leer_parametros_filtrado()
    registros = (DATOS / cfg["carpeta"] / cfg["archivo_registros"]).read_text().split()

    latidos = pd.read_csv(RESULTS / f"qrs_{base}.csv.gz")
    ventanas = pd.read_csv(RESULTS / f"calidad_ventanas_{base}.csv.gz")

    res, resumen = [], []
    for r in registros:
        df_r, resumen_r = procesar_registro(base, r, latidos, ventanas, p, p_filt)
        resumen.extend(resumen_r)
        if not df_r.empty:
            res.append(df_r)

    df_final = pd.concat(res, ignore_index=True)
    salida = RESULTS / f"st_serie_{base}.csv.gz"
    df_final.to_csv(salida, index=False, float_format="%.6g")
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(RESULTS / f"st_serie_resumen_{base}.csv", index=False)

    # Tabla 1: qué pasó con los latidos de las derivaciones procesadas.
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
               "Usados en la serie ST"])
    print("\n ////////// Latidos excluidos de la serie //////////")
    print(tabla_latidos.to_string())

    # Tabla 2: derivaciones omitidas.
    omitidas = resumen[resumen["estado"] != "ok"]
    print("\n ////////// Derivaciones omitidas //////////")
    if omitidas.empty:
        print("Derivaciones omitidas: ninguna")
    else:
        print(omitidas[["registro", "nombre_derivacion", "estado"]]
              .rename(columns={"registro": "Registro", "nombre_derivacion": "Derivación",
                               "estado": "Motivo"})
              .to_string(index=False))

    # Tabla 3: los 5 registros con más latidos no normales.
    por_registro = ok.groupby("registro")[["detectados", "normales"]].sum()
    por_registro["% no normales"] = 100 * (1 - por_registro["normales"] / por_registro["detectados"])
    peores = (por_registro.nlargest(5, "% no normales")
              .rename(columns={"detectados": "Latidos detectados"})
              .drop(columns="normales"))
    peores.index.name = "Registro"
    print("\n ////////// Registros con más latidos no normales //////////")
    print(peores.round(2).to_string())
    carpeta_fig = RESULTS / "figuras"
    carpeta_fig.mkdir(parents=True, exist_ok=True)
    episodios = pd.read_csv(RESULTS / f"episodios_{base}.csv")
    ej = p["ejemplo"]
    figura_serie_st(base, ej["registro"], ej["derivacion"], df_final, latidos,
                    ventanas, episodios, p, p_filt,
                    carpeta_fig / f"st_serie_ejemplo_{base}.png")
    print("Serie ST recuperada. Resultados guardados en st_serie_edb.csv.gz")
    print(">>>>>>>>>>>>>>><<<<<<<<<<<<<<<")