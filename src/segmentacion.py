""" ||||||||||||  7MO PASO: SEGMENTACIÓN POR COEFICIENTE DE DISIMILITUD Y UMBRAL ÚNICO||||||||||||

Detecta cambios sostenidos del nivel de ST en la serie del paso 6 y los
compara con los episodios anotados en la base (pregunta P1).

Método:
    1. En cada punto i de la serie se compara una ventana de referencia (los
       ventana_ref_s anteriores) con una de prueba (los ventana_test_s
       siguientes): eta(i) = |media_prueba - media_ref| / max(desvío_ref,
       desvio_min_mv). eta es máximo justo en las transiciones.
    2. Los límites de segmento son los máximos locales de eta que superan el
       umbral y cambian el nivel al menos cambio_min_mv.
    3. Cada tramo entre límites se clasifica por su nivel respecto de la línea 
       de base de la derivación (elevación, depresión o normal). Un episodio es 
       una racha de tramos consecutivos con el mismo estado no normal y dura al 
       menos duracion_min_episodio_s.
    4. Evaluación: una detección acierta si es del mismo signo que el episodio
       anotado y su inicio está a menos de tolerancia_inicio_s (criterio fijado
       de antemano, emparejamiento uno a uno). El umbral de eta se elige en los
       registros de ajuste y se mide en los de evaluación.

Uso:
    python src/segmentacion.py [base]
Salidas:
    results/segmentacion_<base>.csv             episodios detectados
    results/segmentacion_ajuste_<base>.csv      desempeño de cada umbral (ajuste)
    results/segmentacion_evaluacion_<base>.csv  desempeño final (ajuste y evaluación)
"""
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from rutas import CONFIG, RESULTS

COLUMNAS_DET = ["registro", "derivacion", "inicio_det_s", "fin_det_s",
                "duracion_det_s", "signo", "fin_inferido", "eta_inicio"]


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/segmentacion.json."""
    with open(CONFIG / "segmentacion.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Coeficiente de disimilitud
# ---------------------------------------------------------------------------

def acumulados(x):
    """Sumas acumuladas (con un 0 inicial) de la cantidad de datos válidos, de
    los valores y de sus cuadrados, ignorando los NaN. Permiten calcular la
    media y el desvío de cualquier ventana sin recorrerla."""
    valido = ~np.isnan(x)
    x0 = np.where(valido, x, 0.0)
    cero = np.zeros(1)
    return (np.concatenate([cero, np.cumsum(valido)]),
            np.concatenate([cero, np.cumsum(x0)]),
            np.concatenate([cero, np.cumsum(x0 ** 2)]))


def coeficiente_disimilitud(st_mv, n_ref, n_test, p):
    """Calcula eta y el cambio de nivel en cada punto de la serie.

    Para cada punto i, la referencia son las n_ref muestras anteriores
    [i - n_ref, i) y la prueba las n_test siguientes [i, i + n_test).
        delta(i) = media_prueba - media_referencia      (mV, con signo)
        eta(i)   = |delta| / max(desvío_referencia, desvio_min_mv)
    El piso desvio_min_mv evita que un cambio diminuto dé eta enorme cuando la
    referencia es casi plana. Si una ventana tiene menos de fraccion_valida_min
    de datos válidos, o no entra en la serie, eta y delta quedan en NaN.

    Entradas: st_mv (serie ST, mV, con NaN), n_ref y n_test (muestras) y p.
    Salidas: eta y delta, vectores del largo de st_mv.
    """
    N = len(st_mv)
    eta = np.full(N, np.nan)
    delta = np.full(N, np.nan)
    if N < n_ref + n_test:
        return eta, delta

    n, s1, s2 = acumulados(st_mv)
    i = np.arange(n_ref, N - n_test + 1)
    n_r = n[i] - n[i - n_ref]
    n_t = n[i + n_test] - n[i]
    m_r = (s1[i] - s1[i - n_ref]) / np.maximum(n_r, 1)
    s_r = np.sqrt(np.maximum((s2[i] - s2[i - n_ref]) / np.maximum(n_r, 1) - m_r ** 2, 0.0))
    m_t = (s1[i + n_test] - s1[i]) / np.maximum(n_t, 1)

    suficiente = ((n_r >= n_ref * p["fraccion_valida_min"])
                  & (n_t >= n_test * p["fraccion_valida_min"]))
    dif = m_t - m_r
    delta[i] = np.where(suficiente, dif, np.nan)
    eta[i] = np.where(suficiente, np.abs(dif) / np.maximum(s_r, p["desvio_min_mv"]), np.nan)
    return eta, delta


def preparar_derivacion(registro, derivacion, grupo, p):
    """Calcula eta y delta de una derivación a partir de su serie ST.

    La frecuencia de la grilla se toma de t_s (no se supone 2 Hz).
    Entradas: registro, derivacion, grupo (filas de st_serie de esa derivación)
    y p. Salida: dict con t_s, fs, eta, delta y t_fin_s (último instante con
    dato válido).
    """
    grupo = grupo.sort_values("t_s")
    t_s = grupo["t_s"].to_numpy()
    st = grupo["st_mv"].to_numpy()
    fs = 1.0 / np.median(np.diff(t_s))
    eta, delta = coeficiente_disimilitud(
        st, int(round(p["ventana_ref_s"] * fs)), int(round(p["ventana_test_s"] * fs)), p)
    valido = ~np.isnan(st)
    return {"registro": registro, "derivacion": derivacion, "t_s": t_s, "fs": fs,
            "st_mv": st, "eta": eta, "delta": delta,
            "t_fin_s": t_s[valido][-1] if valido.any() else t_s[-1]}


# ---------------------------------------------------------------------------
# Detección de episodios
# ---------------------------------------------------------------------------

def fila_episodio(d, i_inicio, signo, fin_s, inferido):
    """Arma el dict de un episodio detectado que empieza en el índice i_inicio
    de la serie y termina en fin_s."""
    inicio_s = d["t_s"][i_inicio]
    return {"registro": d["registro"], "derivacion": d["derivacion"],
            "inicio_det_s": inicio_s, "fin_det_s": fin_s,
            "duracion_det_s": fin_s - inicio_s, "signo": signo,
            "fin_inferido": inferido, "eta_inicio": d["eta"][i_inicio]}


def detectar_derivacion(d, umbral, p):
    """Detecta los episodios de una derivación con un umbral de eta.

    1. Límites: máximos locales de eta >= umbral, separados al menos
       separacion_min_limites_s, cuyo cambio de nivel es >= cambio_min_mv.
    2. Los límites dividen la serie en tramos. Cada tramo se clasifica por su
       nivel (mediana de ST) respecto de la línea de base de la derivación (la
       mediana de toda la serie, robusta porque los episodios ocupan una
       minoría del tiempo): elevación si queda desvio_nivel_min_mv o más por
       encima, depresión si queda por debajo, normal en otro caso.
    3. Un episodio es una racha de tramos consecutivos con el mismo estado no
       normal. Empieza en el límite anterior a la racha y termina en el límite
       posterior; si la racha llega al final de la serie, se cierra en el
       último dato válido con fin_inferido=True. Se descartan los de menos de
       duracion_min_episodio_s.
    Usar el nivel y no solo el signo del cambio evita confundir la vuelta a la
    línea de base (un cambio de signo contrario) con el comienzo de un episodio
    opuesto.
    Salida: DataFrame con COLUMNAS_DET (signo "+" elevación, "-" depresión).
    """
    st = d["st_mv"]
    vacio = pd.DataFrame(columns=COLUMNAS_DET)
    if not np.isfinite(st).any():
        return vacio

    eta_lleno = np.nan_to_num(d["eta"], nan=0.0)
    distancia = max(int(round(p["separacion_min_limites_s"] * d["fs"])), 1)
    picos, _ = find_peaks(eta_lleno, height=umbral, distance=distancia)
    picos = picos[np.abs(d["delta"][picos]) >= p["cambio_min_mv"]]

    base = np.nanmedian(st)
    bordes = np.concatenate([[0], picos, [len(st)]]).astype(int)
    estados = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        tramo = st[a:b]
        desvio = np.nanmedian(tramo) - base if np.isfinite(tramo).any() else 0.0
        estados.append(1 if desvio >= p["desvio_nivel_min_mv"]
                       else -1 if desvio <= -p["desvio_nivel_min_mv"] else 0)

    filas, k = [], 0
    while k < len(estados):
        if estados[k] == 0:
            k += 1
            continue
        j = k
        while j + 1 < len(estados) and estados[j + 1] == estados[k]:
            j += 1
        i_inicio = bordes[k] if k > 0 else int(np.argmax(np.isfinite(st)))
        al_final = j == len(estados) - 1
        fin_s = d["t_fin_s"] if al_final else d["t_s"][bordes[j + 1]]
        filas.append(fila_episodio(d, i_inicio, "+" if estados[k] > 0 else "-", fin_s, al_final))
        k = j + 1

    tabla = pd.DataFrame(filas, columns=COLUMNAS_DET)
    return tabla[tabla["duracion_det_s"] >= p["duracion_min_episodio_s"]]

def detectar_todas(datos, umbral, p):
    """Aplica detectar_derivacion a todas las derivaciones de `datos`
    (dict {(registro, derivación): resultado de preparar_derivacion})."""
    tablas = [detectar_derivacion(d, umbral, p) for d in datos.values()]
    tablas = [t for t in tablas if len(t)]
    if not tablas:
        return pd.DataFrame(columns=COLUMNAS_DET)
    return pd.concat(tablas, ignore_index=True)


# ---------------------------------------------------------------------------
# Evaluación contra los episodios anotados
# ---------------------------------------------------------------------------

def episodios_referencia(gt_df, derivaciones, tipo="ST"):
    """Episodios anotados del tipo pedido ("ST" o "T"), sin cambios de eje, en
    las derivaciones que tienen serie. Las derivaciones omitidas en el paso 6
    no se evalúan."""
    ref = gt_df[(gt_df["tipo"] == tipo) & ~gt_df["cambio_eje"]]
    presentes = set(derivaciones)
    en_serie = [(r, d) in presentes for r, d in zip(ref["registro"], ref["derivacion"])]
    return ref[np.array(en_serie, dtype=bool)].reset_index(drop=True)


def emparejar_episodios(det, ref, tolerancia_s):
    """Empareja uno a uno episodios detectados y anotados.

    Un detectado corresponde a un anotado si es de la misma derivación y del
    mismo signo y su inicio está a <= tolerancia_s del inicio anotado. Se
    recorren los anotados por orden de inicio y cada uno toma el detectado
    libre más cercano.
    Salida: lista de pares (índice en ref, índice en det).
    """
    por_derivacion = {k: g for k, g in det.groupby(["registro", "derivacion"])}
    pares = []
    for clave, r in ref.groupby(["registro", "derivacion"]):
        d = por_derivacion.get(clave)
        if d is None:
            continue
        usados = set()
        for idx_ref, fila in r.sort_values("inicio_s").iterrows():
            distancia = (d["inicio_det_s"] - fila["inicio_s"]).abs()
            libre = ((d["signo"] == fila["signo"]) & (distancia <= tolerancia_s)
                     & ~d.index.isin(usados))
            if libre.any():
                idx_det = distancia[libre].idxmin()
                usados.add(idx_det)
                pares.append((idx_ref, idx_det))
    return pares


def evaluar(det, ref, p):
    """Mide el desempeño de las detecciones contra los episodios anotados.

    Recall = aciertos / anotados; Precisión = aciertos / detectados; F1 = media
    armónica. Recall (alcanzables) excluye los anotados que empiezan antes de
    ventana_ref_s, porque ahí todavía no hay ventana de referencia completa.
    Error de inicio y de fin = detectado - anotado (s), solo en los aciertos
    (el de fin, además, solo si ninguno de los dos tiene el fin inferido).
    Salida: dict de métricas.
    """
    pares = emparejar_episodios(det, ref, p["tolerancia_inicio_s"])
    aciertos = len(pares)
    nan = float("nan")
    recall = aciertos / len(ref) if len(ref) else nan
    precision = aciertos / len(det) if len(det) else nan
    f1 = 2 * recall * precision / (recall + precision) if aciertos else nan

    alcanzables = ref["inicio_s"] >= p["ventana_ref_s"]
    acertados = {i for i, _ in pares}
    n_alc = int(alcanzables.sum())
    ac_alc = sum(1 for i in ref.index[alcanzables] if i in acertados)

    error_inicio = np.array([])
    error_fin = np.array([])
    if aciertos:
        r = ref.loc[[i for i, _ in pares]]
        d = det.loc[[j for _, j in pares]]
        error_inicio = d["inicio_det_s"].to_numpy() - r["inicio_s"].to_numpy()
        ambos = ~(d["fin_inferido"].to_numpy(dtype=bool) | r["fin_inferido"].to_numpy(dtype=bool))
        error_fin = (d["fin_det_s"].to_numpy() - r["fin_s"].to_numpy())[ambos]

    def mediana(x):
        return float(np.median(x)) if len(x) else nan

    def ric(x):
        return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else nan

    return {"Episodios anotados": len(ref), "Episodios detectados": len(det),
            "Aciertos": aciertos, "Recall": recall,
            "Recall (alcanzables)": ac_alc / n_alc if n_alc else nan,
            "Precisión": precision, "F1": f1,
            "Error de inicio, mediana (s)": mediana(error_inicio),
            "Error de inicio, RIC (s)": ric(error_inicio),
            "Error de fin, mediana (s)": mediana(error_fin),
            "Error de fin, RIC (s)": ric(error_fin)}


def ajustar_hiperparametros(st_ajuste, ref_ajuste, p):
    """Prueba todas las combinaciones de ventana de referencia, ventana de
    prueba y umbral de eta sobre los registros de ajuste. Las ventanas son
    listas en p (ventanas_ref_s, ventanas_test_s).
    Entradas: st_ajuste (serie ST solo de los registros de ajuste), ref_ajuste
    (episodios anotados de esos registros) y p.
    Salida: DataFrame con una fila por combinación y sus métricas."""
    filas = []
    for v_ref in p["ventanas_ref_s"]:
        for v_test in p["ventanas_test_s"]:
            q = {**p, "ventana_ref_s": v_ref, "ventana_test_s": v_test}
            datos = {(r, d): preparar_derivacion(r, d, g, q)
                     for (r, d), g in st_ajuste.groupby(["registro", "derivacion"])}
            for umbral in q["umbrales_eta"]:
                m = evaluar(detectar_todas(datos, umbral, q), ref_ajuste, q)
                filas.append({"Ventana ref. (s)": v_ref, "Ventana prueba (s)": v_test,
                              "Umbral η": umbral,
                              "Episodios detectados": m["Episodios detectados"],
                              "Recall": m["Recall"], "Precisión": m["Precisión"],
                              "F1": m["F1"]})
    return pd.DataFrame(filas)


def dividir_registros(registros, semilla):
    """Divide los registros al azar (semilla fija) en dos mitades: ajuste y
    evaluación. Limitación: la base tiene 79 sujetos en 90 registros y no se
    dispone del identificador de sujeto, así que un sujeto con dos registros
    puede caer en las dos mitades."""
    mezclados = [str(r) for r in np.random.default_rng(semilla).permutation(sorted(registros))]
    mitad = len(mezclados) // 2
    return set(mezclados[:mitad]), set(mezclados[mitad:])

def figura_segmentacion(st_df, datos, ref, det, registro, derivacion, p, umbral, ruta):
    """Figura de una derivación: serie ST (arriba) y eta (abajo), con los
    episodios anotados (gris), los detectados (naranja) y el umbral de eta.
    Entradas: st_df (serie ST completa), datos (salida de preparar_derivacion
    por derivación), ref y det (episodios anotados y detectados), registro,
    derivacion, p, umbral y ruta del PNG."""
    d = datos[(registro, derivacion)]
    g = st_df[(st_df["registro"] == registro) & (st_df["derivacion"] == derivacion)].sort_values("t_s")
    r = ref[(ref["registro"] == registro) & (ref["derivacion"] == derivacion)]
    q = det[(det["registro"] == registro) & (det["derivacion"] == derivacion)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(g["t_s"] / 60, g["st_mv"], color="navy", lw=0.6)
    ax2.plot(d["t_s"] / 60, d["eta"], color="navy", lw=0.8)
    ax2.axhline(umbral, color="red", ls="--", lw=0.8, label="Umbral η = " + str(umbral))
    for a, b in zip(r["inicio_s"], r["fin_s"]):
        for ax in (ax1, ax2):
            ax.axvspan(a / 60, b / 60, color="gray", alpha=0.25)
    for a, b in zip(q["inicio_det_s"], q["fin_det_s"]):
        for ax in (ax1, ax2):
            ax.axvline(a / 60, color="purple", lw=1.2)
            ax.axvline(b / 60, color="purple", lw=1.2, ls=":")
    ax1.axvspan(0, p["ventana_ref_s"] / 60, color="gray", alpha=0.08)
    ax1.plot([], [], color="gray", alpha=0.4, lw=8, label="Anotado")
    ax1.plot([], [], color="purple", label="Detectado: inicio (—) y fin (···)")
    ax1.set_ylabel("ST (mV)")
    ax2.set_ylabel("η")
    ax2.set_xlabel("Tiempo (min)")
    ax1.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Segmentación por disimilitud, registro " + str(registro) + ", derivación " + str(derivacion))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

COLUMNAS_DET_UMBRAL = ["registro", "derivacion", "inicio_det_s", "fin_det_s",
                       "duracion_det_s", "signo", "fin_inferido", "desvio_max_mv"]


def referencia_st(st_mv, fs, p):
    """Nivel de referencia de ST de una derivación: mediana de los primeros
    referencia_s segundos de datos válidos (la EDB usa como referencia la forma
    de onda de los primeros 30 s del registro). Si hay menos de la mitad de esos
    datos, devuelve NaN."""
    validos = st_mv[np.isfinite(st_mv)]
    n = int(round(p["referencia_s"] * fs))
    return float(np.median(validos[:n])) if len(validos) >= n // 2 else np.nan


def detectar_umbral_derivacion(d, p):
    """Detecta episodios ST de una derivación con los criterios del protocolo de
    anotación de la EDB (PhysioNet, "Annotations"), para elevación y depresión.

    Desvío = ST - referencia (con el signo del episodio).
    1. Núcleo: tramo continuo de al menos duracion_min_s con desvío >= umbral_st_mv.
    2. Dos núcleos son el mismo episodio si entre ellos hay menos de
       duracion_min_s de datos válidos (el desvío no se mantuvo por debajo del
       umbral 30 s).
    3. Inicio: se retrocede desde el primer cruce hasta el último punto con desvío
       < umbral_limite_mv que haya estado < umbral_st_mv en los duracion_min_s
       previos. Fin: igual hacia adelante. Si no hay vuelta, el fin se infiere
       en el último dato válido (fin_inferido=True).
    Los NaN no cuentan como "por debajo": no se puede afirmar que el ST volvió.
    Entrada: d (salida de preparar_derivacion) y p. Salida: DataFrame con
    COLUMNAS_DET_UMBRAL.
    """
    st, fs, t = d["st_mv"], d["fs"], d["t_s"]
    ref = referencia_st(st, fs, p)
    vacio = pd.DataFrame(columns=COLUMNAS_DET_UMBRAL)
    if np.isnan(ref):
        return vacio
    n30 = int(round(p["duracion_min_s"] * fs))
    validos_acum = np.concatenate([[0], np.cumsum(np.isfinite(st))])

    filas = []
    for signo, s in (("+", 1), ("-", -1)):
        v = s * (st - ref)
        arriba = v >= p["umbral_st_mv"]
        cambios = np.diff(np.concatenate([[0], arriba.astype(int), [0]]))
        ini_r, fin_r = np.where(cambios == 1)[0], np.where(cambios == -1)[0] - 1
        largos = (fin_r - ini_r + 1) >= n30
        ini_r, fin_r = ini_r[largos], fin_r[largos]
        if len(ini_r) == 0:
            continue
        grupos = [[ini_r[0], fin_r[0]]]
        for a, b in zip(ini_r[1:], fin_r[1:]):
            if validos_acum[a] - validos_acum[grupos[-1][1] + 1] < n30:
                grupos[-1][1] = b
            else:
                grupos.append([a, b])

        serie = pd.Series(v)
        previo_max = serie.rolling(n30 + 1, min_periods=1).max().to_numpy()
        posterior_max = serie[::-1].rolling(n30 + 1, min_periods=1).max().to_numpy()[::-1]
        claro = v < p["umbral_limite_mv"]
        cand_ini = np.where(claro & (previo_max < p["umbral_st_mv"]))[0]
        cand_fin = np.where(claro & (posterior_max < p["umbral_st_mv"]))[0]
        episodios = []
        for i0, i1 in grupos:
            if p["extender_bordes"]:
                k = np.searchsorted(cand_ini, i0) - 1
                inicio_s = t[cand_ini[k] + 1] if k >= 0 else t[int(np.argmax(np.isfinite(st)))]
                k = np.searchsorted(cand_fin, i1)
                inferido = k >= len(cand_fin)
                fin_s = d["t_fin_s"] if inferido else t[cand_fin[k]]
            else:
                inicio_s = t[i0]
                fin_s = t[i1]
                inferido = bool(i1 >= np.where(np.isfinite(st))[0][-1])
            maximo = np.nanmax(v[i0:i1 + 1])
            if episodios and inicio_s <= episodios[-1]["fin_det_s"]:
                # Dos grupos que quedan en el mismo episodio (justo en el borde de
                # duracion_min_s): se unen.
                previo = episodios[-1]
                previo.update({"fin_det_s": fin_s, "fin_inferido": inferido,
                               "desvio_max_mv": max(previo["desvio_max_mv"], maximo)})
                previo["duracion_det_s"] = previo["fin_det_s"] - previo["inicio_det_s"]
            else:
                episodios.append({"registro": d["registro"], "derivacion": d["derivacion"],
                                  "inicio_det_s": inicio_s, "fin_det_s": fin_s,
                                  "duracion_det_s": fin_s - inicio_s, "signo": signo,
                                  "fin_inferido": inferido, "desvio_max_mv": maximo})
        filas.extend(episodios)

    return pd.DataFrame(filas, columns=COLUMNAS_DET_UMBRAL)


def detectar_umbral_todas(datos, p):
    """Aplica detectar_umbral_derivacion a todas las derivaciones de `datos`."""
    tablas = [detectar_umbral_derivacion(d, p) for d in datos.values()]
    tablas = [t for t in tablas if len(t)]
    return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame(columns=COLUMNAS_DET_UMBRAL)


def fraccion_cubierta(a, b, ini_a, fin_a, ini_b, fin_b):
    """Para cada episodio de `a`, fracción de su duración cubierta por los
    episodios de `b` de la misma derivación y del mismo signo. Las columnas
    ini_*/fin_* son los nombres de las columnas de inicio y fin de cada tabla."""
    grupos = {k: g for k, g in b.groupby(["registro", "derivacion", "signo"])}
    fracciones = []
    for _, fila in a.iterrows():
        g = grupos.get((fila["registro"], fila["derivacion"], fila["signo"]))
        duracion = fila[fin_a] - fila[ini_a]
        if g is None or duracion <= 0:
            fracciones.append(0.0)
            continue
        solape = (np.minimum(g[fin_b], fila[fin_a]) - np.maximum(g[ini_b], fila[ini_a])).clip(lower=0)
        fracciones.append(solape.sum() / duracion)
    return np.array(fracciones)


def evaluar_solapamiento(det, ref):
    """Evaluación por solapamiento, como en ANSI/AAMI EC57 (sección 4.6.4).

    Un episodio anotado es detectado si los detectados de la misma derivación y
    del mismo signo cubren al menos el 50 % de su duración (sensibilidad); un
    detectado es correcto si los anotados de la misma derivación y signo cubren
    al menos el 50 % de su duración (precisión positiva). EC57 acepta también que
    el solapamiento incluya el extremo marcado; como la tabla de anotaciones no
    tiene el instante del extremo, se usa solo la regla del 50 %. Estadísticas
    brutas: se cuentan episodios, sin emparejar uno a uno.
    Salida: dict con anotados, detectados, aciertos de cada lado, Recall,
    Precisión y F1.
    """
    nan = float("nan")
    acierto_ref = int((fraccion_cubierta(ref, det, "inicio_s", "fin_s", "inicio_det_s", "fin_det_s") >= 0.5).sum())
    acierto_det = int((fraccion_cubierta(det, ref, "inicio_det_s", "fin_det_s", "inicio_s", "fin_s") >= 0.5).sum())
    recall = acierto_ref / len(ref) if len(ref) else nan
    precision = acierto_det / len(det) if len(det) else nan
    f1 = 2 * recall * precision / (recall + precision) if acierto_ref and acierto_det else nan
    return {"Episodios anotados": len(ref), "Episodios detectados": len(det),
            "Anotados detectados": acierto_ref, "Detectados correctos": acierto_det,
            "Recall": recall, "Precisión": precision, "F1": f1}

COLORES_UMBRAL = {"serie": "navy", "umbral": "#7fcbb8", "anotado": "gray", "detectado": "#b9a5e6"}

#6f8fb0
def figura_segmentacion_umbral(d, ref, det, p, ruta, nombre = "ST"):
    """Figura de una derivación para el detector por umbral.

    Arriba: serie ST, nivel de referencia, umbrales de ±umbral_st_mv (episodio) y
    ±umbral_limite_mv (inicio y fin) alrededor de la referencia, episodios
    anotados (franjas durazno) y límites de los detectados (líneas lavanda).
    Abajo: línea de tiempo con los episodios anotados y los detectados; el signo
    (+ elevación, - depresión) va dentro de cada barra.
    Entradas: d (salida de preparar_derivacion de esa derivación), ref y det
    (episodios anotados y detectados), p y ruta del PNG."""
    c = COLORES_UMBRAL
    registro, derivacion = d["registro"], d["derivacion"]
    r = ref[(ref["registro"] == registro) & (ref["derivacion"] == derivacion)]
    q = det[(det["registro"] == registro) & (det["derivacion"] == derivacion)]
    nivel = referencia_st(d["st_mv"], d["fs"], p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(d["t_s"] / 60, d["st_mv"], color=c["serie"], lw=0.7)
    ax1.axhline(nivel, color=c["umbral"], lw=1.2, label="Referencia (primeros " + str(p["referencia_s"]) + " s)")
    for signo in (1, -1):
        ax1.axhline(nivel + signo * p["umbral_st_mv"], color=c["umbral"], ls="--", lw=1.2)
        ax1.axhline(nivel + signo * p["umbral_limite_mv"], color=c["umbral"], ls=":", lw=1)
    ax1.plot([], [], color=c["umbral"], ls="--", label="Umbral ±" + str(p["umbral_st_mv"]) + " mV")
    ax1.plot([], [], color=c["umbral"], ls=":", label="Límite ±" + str(p["umbral_limite_mv"]) + " mV")
    for a, b in zip(r["inicio_s"], r["fin_s"]):
        ax1.axvspan(a / 60, b / 60, color=c["anotado"], alpha=0.3, lw=0)
    for a, b in zip(q["inicio_det_s"], q["fin_det_s"]):
        ax1.axvline(a / 60, color=c["detectado"], lw=1.6)
        ax1.axvline(b / 60, color=c["detectado"], lw=1.6, ls=":")
    ax1.plot([], [], color=c["anotado"], lw=8, alpha=0.5, label="Anotado")
    ax1.plot([], [], color=c["detectado"], lw=1.6, label="Detectado: inicio (—) y fin (···)")
    ax1.set_ylabel(nombre +" (mV)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    for fila, (tabla, ini, fin, color) in enumerate((
            (q, "inicio_det_s", "fin_det_s", c["detectado"]), (r, "inicio_s", "fin_s", c["anotado"]))):
        for _, e in tabla.iterrows():
            ax2.broken_barh([(e[ini] / 60, (e[fin] - e[ini]) / 60)], (fila - 0.35, 0.7),
                            facecolors=color, edgecolors="none")
            ax2.text((e[ini] + e[fin]) / 120, fila, e["signo"], ha="center", va="center", fontsize=8, color="#444444")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Detectado", "Anotado"])
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_xlabel("Tiempo (min)")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Detección por umbral de " + nombre +", registro " + str(registro) + ", derivación " + str(derivacion))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ruta, dpi=150)
    plt.close(fig)

def diagnostico_amplitud(datos, ref, p):
    """Para cada episodio anotado, mide cuánto se aparta nuestra serie ST de la
    referencia dentro del episodio (con el signo del episodio).

    Columnas: desvío máximo (mV) y desvío máximo de la mediana móvil de
    duracion_min_s (mV, es lo que hay que sostener durante 30 s). Sirve para ver
    si nuestra serie alcanza siquiera el umbral de la EDB en los episodios
    anotados. Salida: DataFrame con una fila por episodio de ref (los que tienen
    serie)."""
    filas = []
    for _, e in ref.iterrows():
        d = datos.get((e["registro"], e["derivacion"]))
        if d is None:
            continue
        nivel = referencia_st(d["st_mv"], d["fs"], p)
        dentro = (d["t_s"] >= e["inicio_s"]) & (d["t_s"] <= e["fin_s"])
        signo = 1 if e["signo"] == "+" else -1
        v = pd.Series(signo * (d["st_mv"] - nivel))
        n30 = int(round(p["duracion_min_s"] * d["fs"]))
        movil = v.rolling(n30, min_periods=n30 // 2).median().to_numpy()
        vi, mi = v.to_numpy()[dentro], movil[dentro]
        filas.append({"registro": e["registro"], "derivacion": e["derivacion"], "signo": e["signo"],
                      "duracion_s": e["fin_s"] - e["inicio_s"],
                      "desvio_max_mv": np.nanmax(vi) if np.isfinite(vi).any() else np.nan,
                      "desvio_movil_max_mv": np.nanmax(mi) if np.isfinite(mi).any() else np.nan})
    return pd.DataFrame(filas)

if __name__ == "__main__":
    print("\n>>>>>>>>>>>   Paso 7: segmentación por coef de disimilitud vs umbral global en serie ST y T <<<<<<<<<<<<")

    print("\n <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CORRIMIENTO DEL SEGMENTO ST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    base = sys.argv[1] if len(sys.argv) > 1 else "edb"
    p = leer_parametros()
    st_df = pd.read_csv(RESULTS / f"st_serie_{base}.csv.gz")
    gt_df = pd.read_csv(RESULTS / f"episodios_{base}.csv")
    st_df["registro"] = st_df["registro"].astype(str)
    gt_df["registro"] = gt_df["registro"].astype(str)

    ajuste, evaluacion = dividir_registros(set(st_df["registro"].unique()), p["semilla_division"])
    ref_total = episodios_referencia(gt_df, set(zip(st_df["registro"], st_df["derivacion"])))
    ref_ajuste = ref_total[ref_total["registro"].isin(ajuste)]
    ref_evaluacion = ref_total[ref_total["registro"].isin(evaluacion)]

    # Elección de ventanas y umbral en los registros de ajuste.
    tabla_ajuste = ajustar_hiperparametros(st_df[st_df["registro"].isin(ajuste)], ref_ajuste, p)
    mejor = tabla_ajuste.loc[tabla_ajuste["F1"].fillna(0).idxmax()]
    print("\nSEGMENTACIÓN POR COEFICIENTE DE DISIMILITUD")
    print("\n¨¨¨¨¨¨¨¨¨¨¨ Mejores 10 combinaciones en los registros de ajuste (hiperparámetros) ¨¨¨¨¨¨¨¨¨¨¨")
    print(tabla_ajuste.sort_values("F1", ascending=False).head(10).to_string(
        index=False, float_format="%.3f"))
    umbral = mejor["Umbral η"]
    p = {**p, "ventana_ref_s": int(mejor["Ventana ref. (s)"]),
         "ventana_test_s": int(mejor["Ventana prueba (s)"])}
    print("\nElegido: ventana de referencia " + str(p["ventana_ref_s"]) + " s, ventana de prueba "
          + str(p["ventana_test_s"]) + " s, umbral η = " + str(umbral))

    # Detección con los parámetros elegidos y medición en cada mitad.
    datos = {(r, d): preparar_derivacion(r, d, g, p)
             for (r, d), g in st_df.groupby(["registro", "derivacion"])}
    det = detectar_todas(datos, umbral, p)
    det["umbral_eta"] = umbral
    det["conjunto"] = np.where(det["registro"].isin(ajuste), "ajuste", "evaluación")
    tabla_final = pd.DataFrame({
        "Ajuste": evaluar(det[det["conjunto"] == "ajuste"], ref_ajuste, p),
        "Evaluación": evaluar(det[det["conjunto"] == "evaluación"], ref_evaluacion, p),
    }).T

    det.to_csv(RESULTS / f"segmentacion_{base}.csv", index=False, float_format="%.6g")
    tabla_ajuste.to_csv(RESULTS / f"segmentacion_ajuste_{base}.csv", index=False,
                        float_format="%.4g")
    tabla_final.to_csv(RESULTS / f"segmentacion_evaluacion_{base}.csv", float_format="%.4g")

    print("\n¨¨¨¨¨¨¨¨¨¨¨ Desempeño de la segmentación ¨¨¨¨¨¨¨¨¨¨¨")
    mostrar = tabla_final[["Episodios anotados", "Episodios detectados", "Aciertos", "Recall",
                           "Precisión", "F1"]].copy()
    for columna in ["Episodios anotados", "Episodios detectados", "Aciertos"]:
        mostrar[columna] = mostrar[columna].astype(int)
    print(mostrar.to_string(float_format="%.3f"))
    print("\n¨¨¨¨¨¨¨¨¨¨¨ Errores de localización en los aciertos (detectado - anotado) ¨¨¨¨¨¨¨¨¨¨¨")
    print(tabla_final[["Error de inicio, mediana (s)", "Error de inicio, RIC (s)",
                       "Error de fin, mediana (s)", "Error de fin, RIC (s)"]].to_string(
        float_format="%.1f"))

    print("\n¨¨¨¨¨¨¨¨¨¨¨ Primeros 15 episodios detectados ¨¨¨¨¨¨¨¨¨¨¨")
    resumen_det = det[["registro", "derivacion", "inicio_det_s", "fin_det_s", "signo"]].rename(
        columns={"registro": "Registro", "derivacion": "Derivación",
                 "inicio_det_s": "Inicio (s)", "fin_det_s": "Fin (s)", "signo": "Signo"})
    print(resumen_det.head(15).to_string(index=False, float_format="%.0f"))
    print("Total de episodios detectados: " + str(len(det)))

    (RESULTS / "figuras").mkdir(exist_ok=True)
    mas_episodios =  ("e0107", 1)         #Para elegir el de más anotaciones: ref_total.groupby(["registro", "derivacion"]).size().idxmax()
    figura_segmentacion(st_df, datos, ref_total, det, mas_episodios[0], mas_episodios[1], p, umbral,
                        RESULTS / "figuras" / f"segmentacion_ejemplo_{base}.png")
    print(f"\nFIGURA segmentacion_ejemplo_{base}.png: muestra la evolución del coeficiente de "
          "disimilitud, los episodios detectados y los anotados en la derivación con más "
          "episodios de la base.")

    print("SEGMENTACIÓN POR UN ÚNICO UMBRAL DE DESPLAZAMIENTO (criterios del protocolo de la EDB)")
    print(f"\nFIGURA segmentacion_umbral_{base}.png: se ven los episodios detectados por la segmentación con un umbral de desplazamiento\n")
    
    # --- Detección por umbral (criterios del protocolo de la EDB) y comparación ---
    det_umbral = detectar_umbral_todas(datos, p)
    det_umbral.to_csv(RESULTS / f"segmentacion_umbral_{base}.csv", index=False, float_format="%.6g")
    det_eta = det.drop(columns=["umbral_eta", "conjunto"])
    todos_los_registros = set(st_df["registro"])
    comparacion = {}
    for nombre, det_x, conjuntos in (
            ("Disimilitud (η)", det_eta, {"Evaluación": evaluacion}),
            ("Umbral (protocolo EDB)", det_umbral,
             {"Evaluación": evaluacion, "Todos los registros": todos_los_registros})):
        for nombre_conjunto, registros in conjuntos.items():
            comparacion[(nombre, nombre_conjunto)] = evaluar_solapamiento(
                det_x[det_x["registro"].isin(registros)],
                ref_total[ref_total["registro"].isin(registros)])
    tabla_comparacion = pd.DataFrame(comparacion).T
    for columna in ["Episodios anotados", "Episodios detectados", "Anotados detectados",
                    "Detectados correctos"]:
        tabla_comparacion[columna] = tabla_comparacion[columna].astype(int)
    tabla_comparacion.to_csv(RESULTS / f"segmentacion_comparacion_{base}.csv", float_format="%.4g")
    print("\n¨¨¨¨¨¨¨¨¨¨¨ Comparación por solapamiento (criterio EC57) ¨¨¨¨¨¨¨¨¨¨¨")
    print(tabla_comparacion.to_string(float_format="%.3f"))

    figura_segmentacion_umbral(datos[(mas_episodios[0], mas_episodios[1])], ref_total, det_umbral, p,
                               RESULTS / "figuras" / f"segmentacion_umbral_{base}.png")

    diagnostico = diagnostico_amplitud(datos, ref_total, p)
    por_signo = diagnostico.groupby("signo")
    resumen_diagnostico = pd.DataFrame({
        "Episodios": por_signo.size(),
        "Alcanzan el umbral (máximo)": por_signo["desvio_max_mv"].apply(lambda x: (x >= p["umbral_st_mv"]).mean()),
        "Alcanzan el umbral sostenido 30 s": por_signo["desvio_movil_max_mv"].apply(lambda x: (x >= p["umbral_st_mv"]).mean()),
        "Mediana del desvío máximo (mV)": por_signo["desvio_max_mv"].median(),
        "Mediana del desvío sostenido (mV)": por_signo["desvio_movil_max_mv"].median()})
    print("\n¨¨¨¨¨¨¨¨¨¨¨ ¿Alcanza nuestra serie el umbral de 0,1 mV en los episodios anotados? ¨¨¨¨¨¨¨¨¨¨¨")
    print(resumen_diagnostico.to_string(float_format="%.3f"))

    print("\n <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DEFORMACIÓN DE LA ONDA T >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    # --- Detección por umbral en la serie de la onda T (EDB: 0,2 mV durante 30 s) ---
    t_df = pd.read_csv(RESULTS / f"t_serie_{base}.csv.gz").rename(columns={"t_mv": "st_mv"})
    t_df["registro"] = t_df["registro"].astype(str)
    p_t = {**p, "umbral_st_mv": p["umbral_t_mv"], "umbral_limite_mv": p["umbral_limite_t_mv"]}
    datos_t = {(r, d): preparar_derivacion(r, d, g, p_t)
               for (r, d), g in t_df.groupby(["registro", "derivacion"])}
    ref_t = episodios_referencia(gt_df, set(datos_t), "T")
    print("\nEpisodios T anotados por signo: " + str(ref_t["signo"].value_counts().to_dict()))
    det_t = detectar_umbral_todas(datos_t, p_t)
    det_t.to_csv(RESULTS / f"segmentacion_umbral_t_{base}.csv", index=False, float_format="%.6g")

    comparacion_t = {}
    for nombre_conjunto, registros in (("Evaluación", evaluacion),
                                       ("Todos los registros", set(t_df["registro"]))):
        comparacion_t[("Umbral T (protocolo EDB)", nombre_conjunto)] = evaluar_solapamiento(
            det_t[det_t["registro"].isin(registros)],
            ref_t[ref_t["registro"].isin(registros)])
    tabla_comparacion_t = pd.DataFrame(comparacion_t).T
    for columna in ["Episodios anotados", "Episodios detectados", "Anotados detectados",
                    "Detectados correctos"]:
        tabla_comparacion_t[columna] = tabla_comparacion_t[columna].astype(int)
    tabla_comparacion_t.to_csv(RESULTS / f"segmentacion_comparacion_t_{base}.csv", float_format="%.4g")
    print("\n¨¨¨¨¨¨¨¨¨¨¨ Detección de episodios T por solapamiento (criterio EC57) ¨¨¨¨¨¨¨¨¨¨¨")
    print(tabla_comparacion_t.to_string(float_format="%.3f"))

    for nombre, det_x, ref_x in (("ST", det_umbral, ref_total), ("T", det_t, ref_t)):
        m = evaluar(det_x, ref_x, p)
        print(nombre + ": error de inicio mediana " + str(round(m["Error de inicio, mediana (s)"], 1))
              + " s (IQR " + str(round(m["Error de inicio, RIC (s)"], 1)) + "), error de fin mediana "
              + str(round(m["Error de fin, mediana (s)"], 1)) + " s (IQR "
              + str(round(m["Error de fin, RIC (s)"], 1)) + "), aciertos con tolerancia: "
              + str(m["Aciertos"]))

    diagnostico_t = diagnostico_amplitud(datos_t, ref_t, p_t)
    por_signo_t = diagnostico_t.groupby("signo")
    resumen_diagnostico_t = pd.DataFrame({
        "Episodios": por_signo_t.size(),
        "Alcanzan el umbral (máximo)": por_signo_t["desvio_max_mv"].apply(lambda x: (x >= p_t["umbral_st_mv"]).mean()),
        "Alcanzan el umbral sostenido 30 s": por_signo_t["desvio_movil_max_mv"].apply(lambda x: (x >= p_t["umbral_st_mv"]).mean()),
        "Mediana del desvío máximo (mV)": por_signo_t["desvio_max_mv"].median(),
        "Mediana del desvío sostenido (mV)": por_signo_t["desvio_movil_max_mv"].median()})
    print("\n¨¨¨¨¨¨¨¨¨¨¨ ¿Alcanza nuestra serie T el umbral de 0,2 mV en los episodios anotados? ¨¨¨¨¨¨¨¨¨¨¨")
    print(resumen_diagnostico_t.to_string(float_format="%.3f"))

    mas_episodios_t = ref_t.groupby(["registro", "derivacion"]).size().idxmax()
    figura_segmentacion_umbral(datos_t[mas_episodios_t], ref_t, det_t, p_t,
                               RESULTS / "figuras" / f"segmentacion_umbral_t_{base}.png", "T")
    print(f"\nFIGURA segmentacion_umbral_t_{base}.png: detección por umbral en la serie de la onda T "
          "de la derivación con más episodios T anotados.")
    
    afectadas = []
    for (registro, derivacion), d in datos.items():
        n = int(round(p["referencia_s"] * d["fs"]))
        validos = np.where(np.isfinite(d["st_mv"]))[0]
        if len(validos) == 0:
            continue
        t_fin_ref = d["t_s"][validos[min(n, len(validos)) - 1]]
        anotados = ref_total[(ref_total["registro"] == registro) & (ref_total["derivacion"] == derivacion)]
        if (anotados["inicio_s"] < t_fin_ref).any():
            afectadas.append((registro, derivacion, round(float(t_fin_ref))))
    print(str(len(afectadas)) + " derivaciones con un episodio anotado que empieza antes de que termine la ventana de referencia")
    print(afectadas)

    print("Segmentación completada")
    print(">>>>>>>>>>>>>>><<<<<<<<<<<<<<<")