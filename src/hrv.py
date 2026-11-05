""" ||||||||||||  9NO PASO: ANÁLISIS DE HRV ||||||||||||

Variabilidad de la frecuencia cardíaca (HRV) alrededor de los episodios ST.

Para cada episodio ST anotado se mide la HRV en tres ventanas de igual largo
(ventana_s), así las tres condiciones son comparables:
    antes:    ventana_s que terminan guarda_antes_s antes del inicio.
    durante:  los primeros ventana_s del episodio (el episodio debe durar al menos
              ventana_s; si no, se excluye).
    despues:  ventana_s que empiezan guarda_despues_s después del fin.

Intervalos NN: RR entre dos latidos consecutivos, ambos normales (según la
anotación de la base) y fuera de ventanas descartadas por calidad, con RR entre
rr_min_s y rr_max_s y a menos de criterio_ectopico_relativo de la mediana local.
Una ventana es válida si tiene al menos n_nn_min intervalos y una cobertura (suma
de NN / largo de la ventana) de al menos cobertura_min, y no se superpone con
otro episodio anotado de la misma derivación.

Métricas por ventana: FC media, RMSSD, potencia LF y HF (periodograma de
Lomb-Scargle, sin interpolar, con tendencia lineal restada y escalado a ms^2),
cociente LF/HF y LF en unidades normalizadas. No se calcula VLF porque con ventanas
cortas no es interpretable.

Los latidos son los del corazón, no de la derivación: un mismo evento anotado en
varias derivaciones comparte los mismos latidos. Por eso los episodios del mismo
registro y signo con inicios a menos de tolerancia_evento_s se agrupan en un
evento y solo el que tiene más intervalos NN válidos queda como representante.

Uso:
    python src/hrv.py [base] 
Salida:
    results/hrv_<base>.csv    una fila por episodio (columna `representante` marca
                              el que se usa en la estadística)
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from exploracion import mascara_descarte
from rutas import CONFIG, RESULTS
from st_serie import latidos_normales, leer_parametros as leer_parametros_st

CONDICIONES = ["antes", "durante", "despues"]
METRICAS = ["fc_media_lpm", "rmssd_ms", "lf_ms2", "hf_ms2", "lf_hf", "lf_nu", "n_nn", "cobertura"]


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/hrv.json."""
    with open(CONFIG / "hrv.json", encoding="utf-8") as f:
        return json.load(f)


def intervalos_nn(idx_qrs, normales, descartada, p):
    """Intervalos NN de una derivación.

    Un NN es el tiempo entre dos latidos detectados consecutivos, ambos
    normales y fuera de ventanas descartadas (así nunca se salta un latido).
    Se conservan los NN entre rr_min_s y rr_max_s y a menos de
    criterio_ectopico_relativo de la mediana de los 11 NN vecinos.
    Entradas: idx_qrs (muestras), normales y descartada (booleanos; descartada es
    por muestra), p. Salidas: t_nn (s, instante del segundo latido) y nn (s).
    """
    t = idx_qrs / p["fs_hz"]
    valido = normales & ~descartada[idx_qrs]
    pares = valido[:-1] & valido[1:]
    t_nn, nn = t[1:][pares], np.diff(t)[pares]
    en_rango = (nn >= p["rr_min_s"]) & (nn <= p["rr_max_s"])
    t_nn, nn = t_nn[en_rango], nn[en_rango]
    mediana = pd.Series(nn).rolling(11, center=True, min_periods=3).median().to_numpy()
    cercano = np.abs(nn - mediana) <= p["criterio_ectopico_relativo"] * mediana
    return t_nn[cercano], nn[cercano]


def metricas_ventana(t_nn, nn, t0, t1, p):
    """Métricas de HRV en la ventana [t0, t1].

    Si no hay n_nn_min intervalos o la cobertura es menor que cobertura_min,
    las métricas quedan en NaN (n_nn y cobertura se informan igual).
    La potencia espectral se obtiene del periodograma de Lomb-Scargle de los NN
    sin tendencia lineal, escalado por 2*T/N (T = largo de la ventana, N = cantidad
    de NN) para que sea una potencia en ms^2 que no dependa de la FC.
    """
    m = (t_nn >= t0) & (t_nn <= t1)
    t, nn_v = t_nn[m], nn[m]
    salida = {k: np.nan for k in METRICAS}
    salida["n_nn"] = len(nn_v)
    salida["cobertura"] = nn_v.sum() / (t1 - t0)
    if len(nn_v) < p["n_nn_min"] or salida["cobertura"] < p["cobertura_min"]:
        return salida

    consecutivos = np.isclose(np.diff(t), nn_v[1:], atol=1e-3)
    dif = np.diff(nn_v)[consecutivos]
    salida["fc_media_lpm"] = 60.0 / nn_v.mean()
    salida["rmssd_ms"] = 1000 * np.sqrt(np.mean(dif ** 2)) if len(dif) else np.nan

    b = p["bandas_hz"]
    frecs = np.linspace(b["lf"][0], b["hf"][1], 720)
    y = nn_v - np.polyval(np.polyfit(t, nn_v, 1), t)
    psd = lombscargle(t, y, 2 * np.pi * frecs, precenter=False) * 2 * (t1 - t0) / len(t)
    df = frecs[1] - frecs[0]
    lf = 1e6 * psd[(frecs >= b["lf"][0]) & (frecs < b["lf"][1])].sum() * df
    hf = 1e6 * psd[(frecs >= b["hf"][0]) & (frecs <= b["hf"][1])].sum() * df
    salida["lf_ms2"], salida["hf_ms2"] = lf, hf
    salida["lf_hf"] = lf / hf if hf > 0 else np.nan
    salida["lf_nu"] = 100 * lf / (lf + hf) if lf + hf > 0 else np.nan
    return salida


def procesar_episodio(ep, otros, t_nn, nn, p):
    """Métricas de HRV de un episodio en sus tres ventanas.

    Una ventana que se superpone con otro episodio anotado de la misma
    derivación (otros) queda con las métricas en NaN. Si el episodio dura menos
    que ventana_s, se devuelve None.
    Salida: dict (una fila de la tabla de salida) o None.
    """
    L, ini, fin = p["ventana_s"], ep["inicio_s"], ep["fin_s"]
    if fin - ini < L:
        return None
    ventanas = {"antes": (ini - p["guarda_antes_s"] - L, ini - p["guarda_antes_s"]),
                "durante": (ini, ini + L),
                "despues": (fin + p["guarda_despues_s"], fin + p["guarda_despues_s"] + L)}
    fila = {"registro": ep["registro"], "derivacion": ep["derivacion"],
            "signo": ep["signo"], "inicio_s": ini, "fin_s": fin}
    for nombre, (t0, t1) in ventanas.items():
        contaminada = ((otros["fin_s"] > t0) & (otros["inicio_s"] < t1)).any() and nombre != "durante"
        met = metricas_ventana(t_nn, nn, t0, t1, p)
        if contaminada:
            met.update({k: np.nan for k in METRICAS if k not in ("n_nn", "cobertura")})
        for k, v in met.items():
            fila[k + "_" + nombre] = v
    return fila


def marcar_eventos(tabla, tolerancia_s):
    """Agrupa episodios del mismo registro y signo con inicios encadenados a
    menos de tolerancia_s y marca como representante el de más NN válidos.
    Agrega las columnas evento_id y representante."""
    tabla = tabla.sort_values(["registro", "signo", "inicio_s"]).reset_index(drop=True)
    nuevo = ((tabla["registro"] != tabla["registro"].shift()) | (tabla["signo"] != tabla["signo"].shift())
             | (tabla["inicio_s"].diff() > tolerancia_s))
    tabla["evento_id"] = nuevo.cumsum()
    total_nn = tabla[["n_nn_" + c for c in CONDICIONES]].sum(axis=1)
    mejor = total_nn.groupby(tabla["evento_id"]).idxmax()
    tabla["representante"] = tabla.index.isin(mejor)
    return tabla


if __name__ == "__main__":
    print("\n>>>>>>>>>>>   Paso 9: HRV alrededor de los episodios ST   <<<<<<<<<<<<")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    p = leer_parametros()
    p_st = leer_parametros_st()
    qrs_df = pd.read_csv(RESULTS / f"qrs_{base}.csv.gz")
    gt_df = pd.read_csv(RESULTS / f"episodios_{base}.csv")
    ventanas_calidad = pd.read_csv(RESULTS / f"calidad_ventanas_{base}.csv.gz")
    gt_st = gt_df[(gt_df["tipo"] == "ST") & ~gt_df["cambio_eje"]].reset_index(drop=True)

    nn_por_derivacion = {}
    for (registro, derivacion), g in qrs_df.groupby(["registro", "derivacion"]):
        idx_qrs = np.sort(g["idx_qrs"].to_numpy())
        normales = latidos_normales(base, registro, idx_qrs, p["fs_hz"], p_st)
        descartada = mascara_descarte(ventanas_calidad, registro, derivacion, idx_qrs[-1] + 1, p["fs_hz"])
        nn_por_derivacion[(registro, derivacion)] = intervalos_nn(idx_qrs, normales, descartada, p)

    filas, sin_nn, cortos = [], 0, 0
    for _, ep in gt_st.iterrows():
        clave = (ep["registro"], ep["derivacion"])
        if clave not in nn_por_derivacion or len(nn_por_derivacion[clave][0]) < p["n_nn_min"]:
            sin_nn += 1
            continue
        otros = gt_st[(gt_st["registro"] == ep["registro"]) & (gt_st["derivacion"] == ep["derivacion"])
                      & (gt_st.index != ep.name)]
        fila = procesar_episodio(ep, otros, *nn_por_derivacion[clave], p)
        if fila is None:
            cortos += 1
        else:
            filas.append(fila)

    tabla = marcar_eventos(pd.DataFrame(filas), p["tolerancia_evento_s"])
    cols_vent = [k + "_" + c for c in CONDICIONES for k in ("lf_hf",)]
    tabla["completo"] = tabla[cols_vent].notna().all(axis=1)
    tabla.to_csv(RESULTS / f"hrv_{base}.csv", index=False, float_format="%.6g")

    resumen = pd.DataFrame({
        "Episodios anotados": gt_st.groupby("signo").size(),
        "Sin NN suficientes en la derivación": pd.Series(
            gt_st[[(r, d) not in nn_por_derivacion or len(nn_por_derivacion[(r, d)][0]) < p["n_nn_min"]
                   for r, d in zip(gt_st["registro"], gt_st["derivacion"])]].groupby("signo").size()),
        "Con tres ventanas válidas": tabla[tabla["completo"]].groupby("signo").size(),
        "Eventos (representantes completos)": tabla[tabla["completo"] & tabla["representante"]].groupby("signo").size(),
    }).fillna(0).astype(int)
    resumen["Duran menos que la ventana"] = gt_st.assign(d=gt_st["fin_s"] - gt_st["inicio_s"]).query(
        "d < @p['ventana_s']").groupby("signo").size().reindex(resumen.index).fillna(0).astype(int)
    resumen.index = resumen.index.map({"+": "Elevación de ST", "-": "Depresión de ST"})
    print("\n####### Episodios incluidos y excluidos #######")
    print(resumen.to_string())
    print(f"\nGuardado en hrv_{base}.csv")