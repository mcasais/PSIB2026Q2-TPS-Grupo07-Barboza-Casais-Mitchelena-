"""Filtrado de la línea de base y del ruido, y su efecto sobre el ST (paso 4).

Banco de filtros (todos de fase cero, para no correr ni deformar el ST):
    pa_005    pasaaltos Butterworth de 0,05 Hz. Es el corte que recomienda
              Kligfield 2007 (AHA) para no distorsionar el ST.
    pa_067    pasaaltos de 0,67 Hz. Kligfield lo admite solo con fase cero;
              se incluye como comparación, para ver cuánto ST se pierde.
    splines   línea de base por splines cúbicas con un nodo por latido en el
              tramo isoeléctrico del PR, que se resta a la señal.
    wiener    filtro óptimo de Wiener con el latido promedio como modelo de
              señal (Rangayyan, cap. 3): H(f) = Ss / (Ss + Sn).
    notch_50  ranura de 50 Hz (interferencia de línea, fuerte en e0147).
    pb_40     pasabajos de 40 Hz (ruido muscular).
Los filtros se combinan en 'cadenas' (config/filtrado.json); p. ej.
'wiener' = pa_005 seguido de Wiener.

Los filtros por splines y Wiener necesitan las posiciones de los latidos:
se usan las de Pan-Tompkins (src/qrs.py, results/qrs_<base>.csv.gz).

Evaluación del efecto sobre el ST:
    1. Respuesta en frecuencia y prueba de impulso de la AHA/IEC (filtros
       lineales): un pulso de 3 mV x 100 ms no debe dejar un desplazamiento
       mayor a 0,1 mV ni una pendiente mayor a 0,3 mV/s.
    2. Señal sintética: latidos reales promediados, con un episodio de ST de
       amplitud conocida, deriva, ruido blanco y 50 Hz. Se compara el ST
       medido después de cada cadena con el verdadero.
    3. Señal real: desviación del ST en el pico de cada episodio anotado,
       medida después de cada cadena, contra la desviación anotada en la base;
       y ruido latido a latido de la serie ST fuera de los episodios.

Medición del ST usada acá (PROVISORIA, la reemplaza el paso 6): nivel en
R + 120 ms (aprox. J + 80 ms) menos el nivel isoeléctrico del PR, ambos
promediados en 20 ms. Alcanza para comparar filtros entre sí.

Uso (desde la raíz del repo, después de src/qrs.py):
    python src/filtrado.py [base] [registro ...]
Salidas:
    results/filtrado_impulso_<base>.csv
    results/filtrado_sintetica_<base>.csv
    results/filtrado_real_episodios_<base>.csv
    results/filtrado_real_ruido_<base>.csv
    results/figuras/filtrado_*.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')                 # solo se guardan figuras, sin ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import (butter, fftconvolve, filtfilt, freqz, iirnotch,
                          sosfiltfilt, sosfreqz)
from scipy.signal.windows import tukey
from scipy.stats import spearmanr

from episodios import leer_config
from exploracion import COLOR_1, COLOR_2, mascara_descarte
from qrs import leer_senal_mv
from rutas import CONFIG, DATOS, RESULTS


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/filtrado.json."""
    with open(CONFIG / 'filtrado.json', encoding='utf-8') as f:
        return json.load(f)


def ms_a_muestras(t_ms, fs):
    """Convierte milisegundos a número (entero) de muestras."""
    return int(round(t_ms / 1000 * fs))


# ---------------------------------------------------------------------------
# Puntos isoeléctricos y medición del ST
# ---------------------------------------------------------------------------

def nivel_isoelectrico(x_mv, fs, idx_qrs, p):
    """Busca en cada latido el tramo más plano del segmento PR.

    Dentro de la ventana de búsqueda (R − 120 a R − 40 ms) se elige el tramo
    de 20 ms con menor suma de |diferencias entre muestras|, o sea el más
    plano: ahí la señal está en la línea isoeléctrica y no en la onda P ni
    en el inicio del QRS.

    Entradas: x_mv (mV), fs (Hz), idx_qrs (muestras), p (parámetros).
    Salidas: idx_iso (muestra central del tramo elegido), nivel_iso_mv
    (promedio del tramo) y validos (máscara sobre idx_qrs de los latidos
    cuya ventana entra en la señal).
    """
    ini = ms_a_muestras(p['isoelectrico']['busqueda_ms'][0], fs)
    fin = ms_a_muestras(p['isoelectrico']['busqueda_ms'][1], fs)
    n_ancho = ms_a_muestras(p['isoelectrico']['ancho_ms'], fs)
    validos = (idx_qrs + ini >= 0) & (idx_qrs + fin <= len(x_mv))
    idx = idx_qrs[validos]

    # Matriz latidos x muestras de la ventana de búsqueda.
    tramos = x_mv[idx[:, None] + np.arange(ini, fin)]
    variacion = np.abs(np.diff(tramos, axis=1))
    # Suma móvil de n_ancho − 1 diferencias = variación dentro de cada tramo
    # de n_ancho muestras (suma acumulada y resta, para no hacer un bucle).
    acumulada = np.concatenate([np.zeros((len(idx), 1)),
                                np.cumsum(variacion, axis=1)], axis=1)
    suma_tramo = acumulada[:, n_ancho - 1:] - acumulada[:, :-(n_ancho - 1)]
    inicio_tramo = np.argmin(suma_tramo, axis=1)

    columnas = inicio_tramo[:, None] + np.arange(n_ancho)
    nivel_iso_mv = np.take_along_axis(tramos, columnas, axis=1).mean(axis=1)
    idx_iso = idx + ini + inicio_tramo + n_ancho // 2
    return idx_iso, nivel_iso_mv, validos


def medir_st(x_mv, fs, idx_qrs, p):
    """Nivel del ST de cada latido respecto del PR (medición provisoria).

    ST = promedio de 20 ms centrado en R + 120 ms − nivel isoeléctrico.
    Entradas: x_mv (mV), fs (Hz), idx_qrs (muestras), p (parámetros).
    Salida: DataFrame con idx_qrs y st_mv (solo latidos cuyas ventanas
    entran en la señal).
    """
    _, nivel_iso_mv, validos = nivel_isoelectrico(x_mv, fs, idx_qrs, p)
    idx = idx_qrs[validos]
    centro = ms_a_muestras(p['medicion_st']['desde_r_ms'], fs)
    medio = ms_a_muestras(p['medicion_st']['ancho_ms'], fs) // 2
    dentro = idx + centro + medio < len(x_mv)
    columnas = idx[dentro, None] + np.arange(centro - medio, centro + medio + 1)
    st_mv = x_mv[columnas].mean(axis=1) - nivel_iso_mv[dentro]
    return pd.DataFrame({'idx_qrs': idx[dentro], 'st_mv': st_mv})


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def filtro_lineal(cfg, fs):
    """Diseña un filtro lineal del banco.

    Entradas: cfg (dict del filtro en config/filtrado.json), fs (Hz).
    Salida: ('sos', matriz sos) para los Butterworth o ('ba', (b, a)) para
    la ranura. Se diseñan en secciones de segundo orden (sos) porque con un
    corte de 0,05 Hz a 250 Hz los coeficientes b, a quedan mal condicionados.
    """
    if cfg['tipo'] == 'pasaaltos':
        return 'sos', butter(cfg['orden'], cfg['corte_hz'], btype='highpass',
                             fs=fs, output='sos')
    if cfg['tipo'] == 'pasabajos':
        return 'sos', butter(cfg['orden'], cfg['corte_hz'], btype='lowpass',
                             fs=fs, output='sos')
    if cfg['tipo'] == 'notch':
        return 'ba', iirnotch(cfg['frecuencia_hz'], cfg['q'], fs=fs)
    raise ValueError(f'{cfg["tipo"]} no es un filtro lineal')


def aplicar_lineal(x_mv, fs, cfg, p):
    """Aplica un filtro lineal ida y vuelta (fase cero).

    filtfilt filtra hacia adelante y hacia atrás: la fase se cancela (no hay
    retardo ni deformación por fase) y el módulo queda al cuadrado (el corte
    a −3 dB de cada pasada queda a −6 dB). Para que el transitorio del
    comienzo no caiga sobre la señal, se extiende la señal en los bordes
    (reflexión impar) durante `padding_periodos` períodos del corte.
    """
    forma, coef = filtro_lineal(cfg, fs)
    frecuencia = cfg.get('corte_hz', cfg.get('frecuencia_hz'))
    padlen = min(int(round(p['padding_periodos'] * fs / frecuencia)), len(x_mv) - 1)
    if forma == 'sos':
        return sosfiltfilt(coef, x_mv, padlen=padlen)
    b, a = coef
    return filtfilt(b, a, x_mv, padlen=padlen)


def linea_base_splines(x_mv, fs, idx_qrs, descartada, p):
    """Estima la línea de base con splines cúbicas por los puntos del PR.

    Un nodo por latido, en el tramo isoeléctrico (nivel_isoelectrico). Se
    excluyen los latidos en ventanas descartadas por calidad, para no poner
    nodos sobre ruido. Antes del primer nodo y después del último se
    mantiene constante el valor del nodo extremo.
    Salida: línea de base (mV), del mismo largo que x_mv.
    """
    idx_iso, nivel_iso_mv, _ = nivel_isoelectrico(x_mv, fs, idx_qrs, p)
    usar = ~descartada[idx_iso]
    idx_iso, nivel_iso_mv = idx_iso[usar], nivel_iso_mv[usar]
    spline = CubicSpline(idx_iso, nivel_iso_mv, bc_type='natural')
    t = np.clip(np.arange(len(x_mv)), idx_iso[0], idx_iso[-1])
    return spline(t)


def wiener_latido_promedio(x_mv, fs, idx_qrs, descartada, p):
    """Filtro de Wiener con el latido promedio como modelo de la señal.

    Por tramos de 60 s:
      - se cortan los latidos (R − 250 a R + 450 ms) y se promedian;
      - Sn(f): espectro medio de los residuos (latido − promedio), que es lo
        que cambia de latido a latido, o sea el ruido;
      - Ss(f): espectro del promedio menos Sn/N, porque el promedio de N
        latidos todavía conserva una varianza de ruido Sn/N;
      - H(f) = Ss / (Ss + Sn), real y entre 0 y 1: fase cero.
    H se pasa a una respuesta al impulso simétrica (FIR por muestreo en
    frecuencia, con ventana de Hann para suavizar el truncamiento) y se
    convoluciona con el tramo (más un margen de los tramos vecinos para que
    los bordes no queden mal filtrados). Los tramos con menos de
    `min_latidos` latidos válidos quedan sin filtrar.

    Entradas: x_mv (mV), fs (Hz), idx_qrs (muestras), descartada (máscara
    por muestra), p. Salidas: señal filtrada (mV) y cantidad de tramos que
    quedaron sin filtrar.
    """
    w = p['wiener']
    ini = ms_a_muestras(w['ventana_latido_ms'][0], fs)
    fin = ms_a_muestras(w['ventana_latido_ms'][1], fs)
    nfft = w['nfft']
    n_segmento = int(round(w['segmento_s'] * fs))
    ventana_fir = np.hanning(nfft - 1)

    y = x_mv.copy()
    sin_filtrar = 0
    for s0 in range(0, len(x_mv), n_segmento):
        s1 = min(s0 + n_segmento, len(x_mv))
        idx = idx_qrs[(idx_qrs >= s0) & (idx_qrs < s1)]
        idx = idx[(idx + ini >= 0) & (idx + fin <= len(x_mv))]
        idx = idx[~descartada[idx]]
        if len(idx) < w['min_latidos']:
            sin_filtrar += 1
            continue

        latidos = x_mv[idx[:, None] + np.arange(ini, fin)]
        promedio = latidos.mean(axis=0)
        s_promedio = np.abs(np.fft.rfft(promedio, nfft)) ** 2
        s_ruido = np.mean(np.abs(np.fft.rfft(latidos - promedio, nfft, axis=1)) ** 2, axis=0)
        s_senal = np.clip(s_promedio - s_ruido / len(idx), 0, None)
        h_f = s_senal / (s_senal + s_ruido + 1e-12)

        # irfft da la respuesta con el retardo 0 en la posición 0; fftshift lo
        # lleva al centro (posición nfft/2). Se descarta la primera muestra
        # para que el largo sea impar y el centro quede exactamente en el
        # medio: así mode='same' no corre la señal.
        h = np.fft.fftshift(np.fft.irfft(h_f, nfft))[1:] * ventana_fir

        a0, a1 = max(s0 - nfft, 0), min(s1 + nfft, len(x_mv))
        y_tramo = fftconvolve(x_mv[a0:a1], h, mode='same')
        y[s0:s1] = y_tramo[s0 - a0:s1 - a0]
    return y, sin_filtrar


def aplicar_cadena(x_mv, fs, idx_qrs, descartada, nombre_cadena, p):
    """Aplica en orden los filtros de una cadena de config/filtrado.json.

    Entradas: x_mv (señal cruda de una derivación, mV), fs (Hz), idx_qrs
    (latidos de Pan-Tompkins, muestras), descartada (máscara por muestra de
    las ventanas descartadas), nombre de la cadena y p.
    Salida: señal filtrada (mV).
    """
    y = x_mv
    for nombre_filtro in p['cadenas'][nombre_cadena]:
        cfg = p['filtros'][nombre_filtro]
        if cfg['tipo'] == 'splines':
            y = y - linea_base_splines(y, fs, idx_qrs, descartada, p)
        elif cfg['tipo'] == 'wiener':
            y, _ = wiener_latido_promedio(y, fs, idx_qrs, descartada, p)
        else:
            y = aplicar_lineal(y, fs, cfg, p)
    return y


# ---------------------------------------------------------------------------
# 1. Respuesta en frecuencia y prueba de impulso
# ---------------------------------------------------------------------------

def respuesta_fase_cero(cfg, fs, f_hz):
    """|H(f)|² del filtro aplicado ida y vuelta, en las frecuencias f_hz."""
    forma, coef = filtro_lineal(cfg, fs)
    if forma == 'sos':
        _, h = sosfreqz(coef, worN=f_hz, fs=fs)
    else:
        _, h = freqz(*coef, worN=f_hz, fs=fs)
    return np.abs(h) ** 2


def prueba_impulso(nombre_filtro, fs, p):
    """Prueba de impulso de la AHA/IEC para un filtro lineal.

    Un pulso rectangular de 3 mV y 100 ms (0,3 mV·s) en el medio de 60 s de
    señal nula. Fuera del pulso, la salida no debería apartarse más de
    0,1 mV de cero ni tener pendiente mayor a 0,3 mV/s. Con fase cero la
    respuesta es simétrica, así que se mide antes y después del pulso.
    Salidas: dict con el resultado y (t_s, salida) para la figura.
    """
    imp = p['impulso']
    n = int(60 * fs)
    x = np.zeros(n)
    i0, i1 = n // 2, n // 2 + int(round(imp['duracion_s'] * fs))
    x[i0:i1] = imp['amplitud_mv']
    y = aplicar_lineal(x, fs, p['filtros'][nombre_filtro], p)

    # La derivada centrada de las muestras pegadas al pulso todavía incluye
    # el salto de 3 mV, así que se deja una muestra de guarda a cada lado.
    fuera = np.ones(n, dtype=bool)
    fuera[i0 - 1:i1 + 1] = False
    pendiente = np.gradient(y) * fs                          # mV/s
    desplazamiento = np.abs(y[fuera]).max()
    pendiente_max = np.abs(pendiente[fuera]).max()
    resultado = {
        'filtro': nombre_filtro,
        'desplazamiento_max_mv': desplazamiento,
        'pendiente_max_mv_s': pendiente_max,
        'cumple': (desplazamiento <= imp['desplazamiento_max_mv'])
                  and (pendiente_max <= imp['pendiente_max_mv_s']),
    }
    return resultado, (np.arange(n) / fs - i0 / fs, y)


# ---------------------------------------------------------------------------
# 2. Señal sintética
# ---------------------------------------------------------------------------

def plantilla_limpia(base, registro, derivacion, latidos, ventanas, episodios, p):
    """Latido promedio de un minuto limpio de un registro real.

    Se busca el primer minuto sin ventanas descartadas y sin episodios
    anotados en esa derivación; se filtra con pa_005 y se promedian los
    latidos (ventana del Wiener). Se le resta el nivel isoeléctrico y se
    multiplica por una ventana de Tukey para que empiece y termine en cero
    (así los latidos se pueden pegar uno detrás de otro sin escalones).
    Salidas: plantilla (mV) y posición del R dentro de ella (muestras).
    """
    x_mv, fs, _ = leer_senal_mv(base, registro)
    x = aplicar_lineal(x_mv[:, derivacion], fs, p['filtros']['pa_005'], p)
    descartada = mascara_descarte(ventanas, registro, derivacion, len(x), fs)
    eps = episodios[(episodios['registro'] == registro)
                    & (episodios['derivacion'] == derivacion)]
    idx_qrs = latidos.loc[(latidos['registro'] == registro)
                          & (latidos['derivacion'] == derivacion), 'idx_qrs'].to_numpy()

    n_min = int(60 * fs)
    for s0 in range(0, len(x) - n_min, n_min):
        t0, t1 = s0 / fs, (s0 + n_min) / fs
        con_episodio = ((eps['inicio_s'] < t1) & (eps['fin_s'] > t0)).any()
        if descartada[s0:s0 + n_min].any() or con_episodio:
            continue
        ini = ms_a_muestras(p['wiener']['ventana_latido_ms'][0], fs)
        fin = ms_a_muestras(p['wiener']['ventana_latido_ms'][1], fs)
        idx = idx_qrs[(idx_qrs >= s0 - ini) & (idx_qrs < s0 + n_min - fin)]
        plantilla = x[idx[:, None] + np.arange(ini, fin)].mean(axis=0)
        _, nivel, _ = nivel_isoelectrico(plantilla, fs, np.array([-ini]), p)
        return (plantilla - nivel[0]) * tukey(len(plantilla), 0.2), -ini
    raise ValueError(f'{registro}: no hay un minuto limpio sin episodios')


def senal_sintetica(plantilla, idx_r, fs, p, rng):
    """Arma un ECG sintético con un episodio de ST conocido.

    - Latidos: la plantilla repetida con RR = 0,85 s ± 0,05 s (variación
      sinusoidal a 0,25 Hz, como una arritmia respiratoria).
    - Episodio: a cada latido se le suma un escalón suave (Tukey) entre
      R + 40 ms y R + 400 ms, cuya amplitud sigue un trapecio en el tiempo
      (0 → −0,2 mV en 60 s, meseta de 120 s, vuelta a 0 en 60 s).
    - Ruido (solo en la versión con ruido): deriva = suma de sinusoides de
      0,01 a 0,5 Hz con amplitud ∝ 1/f y fase al azar, escalada a 0,3 mV
      eficaces; ruido blanco de 0,02 mV; interferencia de 50 Hz de 0,05 mV.

    Salidas: dict con x_limpia (mV), x_ruidosa (mV), idx_qrs (muestras),
    desplazamiento_mv (amplitud del episodio en cada latido) y t_s.
    """
    s = p['sintetica']
    n = int(s['duracion_s'] * fs)
    t_s = np.arange(n) / fs

    # Posiciones de los R.
    r_s = [1.0]
    while r_s[-1] < s['duracion_s'] - 1.0:
        rr = s['rr_medio_s'] + s['rr_amplitud_s'] * np.sin(
            2 * np.pi * s['rr_frecuencia_hz'] * r_s[-1])
        r_s.append(r_s[-1] + rr)
    idx_qrs = np.round(np.array(r_s[:-1]) * fs).astype(int)

    # Amplitud del episodio (trapecio) en el instante de cada latido.
    ep = s['episodio']
    t_r = idx_qrs / fs
    subida = np.clip((t_r - ep['inicio_s']) / ep['rampa_s'], 0, 1)
    t_bajada = ep['inicio_s'] + ep['rampa_s'] + ep['meseta_s']
    bajada = np.clip((t_r - t_bajada) / ep['rampa_s'], 0, 1)
    desplazamiento_mv = ep['amplitud_mv'] * (subida - bajada)

    d0 = ms_a_muestras(s['desplazamiento_st_ms'][0], fs)
    d1 = ms_a_muestras(s['desplazamiento_st_ms'][1], fs)
    escalon = tukey(d1 - d0, 0.2)
    x_limpia = np.zeros(n)
    for idx, amp in zip(idx_qrs, desplazamiento_mv):
        x_limpia[idx - idx_r:idx - idx_r + len(plantilla)] += plantilla
        x_limpia[idx + d0:idx + d1] += amp * escalon

    # Ruido.
    der = s['deriva']
    f_hz = np.exp(rng.uniform(np.log(der['f_min_hz']), np.log(der['f_max_hz']),
                              der['n_componentes']))
    fases = rng.uniform(0, 2 * np.pi, der['n_componentes'])
    deriva = sum(np.sin(2 * np.pi * f * t_s + fi) / f for f, fi in zip(f_hz, fases))
    deriva *= der['rms_mv'] / np.sqrt(np.mean(deriva ** 2))
    ruido = (deriva + s['ruido_blanco_mv'] * rng.standard_normal(n)
             + s['linea_50hz_mv'] * np.sin(2 * np.pi * 50 * t_s + rng.uniform(0, 2 * np.pi)))
    return {'x_limpia': x_limpia, 'x_ruidosa': x_limpia + ruido,
            'idx_qrs': idx_qrs, 'desplazamiento_mv': desplazamiento_mv, 't_s': t_s}


def evaluar_sintetica(base, latidos, ventanas, episodios, fs, p):
    """Mide el ST con cada cadena sobre señales sintéticas.

    ST verdadero = ST medido en la señal limpia sin filtrar (incluye el
    nivel propio de la plantilla más el desplazamiento del episodio). Se
    descartan los primeros y últimos 20 s, donde actúan los bordes.
    Salida: DataFrame con una fila por (plantilla, repetición, condición,
    cadena): sesgo y RMSE del ST latido a latido, y error del cambio del ST
    durante la meseta del episodio respecto de antes del episodio (mV).
    """
    s = p['sintetica']
    ep = s['episodio']
    filas = []
    for registro, derivacion in s['plantillas']:
        plantilla, idx_r = plantilla_limpia(base, registro, derivacion, latidos,
                                            ventanas, episodios, p)
        for rep in range(s['repeticiones']):
            rng = np.random.default_rng(rep)
            sint = senal_sintetica(plantilla, idx_r, fs, p, rng)
            verdadero = medir_st(sint['x_limpia'], fs, sint['idx_qrs'], p)
            t_r = verdadero['idx_qrs'] / fs
            en_margen = (t_r > s['margen_s']) & (t_r < s['duracion_s'] - s['margen_s'])
            antes = en_margen & (t_r < ep['inicio_s'])
            meseta = ((t_r > ep['inicio_s'] + ep['rampa_s'])
                      & (t_r < ep['inicio_s'] + ep['rampa_s'] + ep['meseta_s']))
            nada_descartado = np.zeros(len(sint['x_limpia']), dtype=bool)

            for condicion in ('x_limpia', 'x_ruidosa'):
                for cadena in p['cadenas']:
                    y = aplicar_cadena(sint[condicion], fs, sint['idx_qrs'],
                                       nada_descartado, cadena, p)
                    medido = medir_st(y, fs, sint['idx_qrs'], p)
                    error = (medido['st_mv'] - verdadero['st_mv']).to_numpy()
                    cambio = (np.median(medido['st_mv'][meseta])
                              - np.median(medido['st_mv'][antes]))
                    filas.append({
                        'plantilla': f'{registro}-{derivacion}',
                        'repeticion': rep,
                        'condicion': 'sin_ruido' if condicion == 'x_limpia' else 'con_ruido',
                        'cadena': cadena,
                        'sesgo_mv': error[en_margen].mean(),
                        'rmse_mv': np.sqrt(np.mean(error[en_margen] ** 2)),
                        'error_cambio_mv': cambio - ep['amplitud_mv'],
                    })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 3. Señal real
# ---------------------------------------------------------------------------

def evaluar_real(base, registro, latidos, ventanas, episodios, p):
    """Mide la serie ST de un registro con cada cadena.

    Para cada episodio ST anotado (sin cambios de eje) en la derivación:
    desviación medida = mediana del ST en ±5 s alrededor del pico anotado
    menos la mediana de los primeros 30 s válidos del registro. Así se
    define la desviación anotada en la EDB (respecto del comienzo del
    registro), y se compara con desviacion_max_mv.
    Ruido de la serie: desvío robusto (1,4826·MAD) de la diferencia entre
    latidos consecutivos, dividido por √2, fuera de todo episodio.
    Siempre se excluyen los latidos en ventanas descartadas.
    Salidas: (lista de dicts por episodio, lista de dicts de ruido).
    """
    r = p['real']
    x_mv, fs, nombres = leer_senal_mv(base, registro)
    filas_ep, filas_ruido = [], []
    for k, nombre in enumerate(nombres):
        x = x_mv[:, k]
        idx_qrs = latidos.loc[(latidos['registro'] == registro)
                              & (latidos['derivacion'] == k), 'idx_qrs'].to_numpy()
        descartada = mascara_descarte(ventanas, registro, k, len(x), fs)
        eps = episodios[(episodios['registro'] == registro)
                        & (episodios['derivacion'] == k)]
        eps_st = eps[(eps['tipo'] == 'ST') & ~eps['cambio_eje']
                     & eps['desviacion_max_mv'].notna()]

        for cadena in p['cadenas']:
            y = aplicar_cadena(x, fs, idx_qrs, descartada, cadena, p)
            st = medir_st(y, fs, idx_qrs, p)
            st = st[~descartada[st['idx_qrs']]]
            t_s = st['idx_qrs'].to_numpy() / fs
            valor = st['st_mv'].to_numpy()

            if len(t_s) == 0:
                continue
            referencia = np.median(valor[t_s < t_s[0] + r['referencia_s']])
            for _, ep in eps_st.iterrows():
                cerca = np.abs(t_s - ep['pico_s']) <= r['ventana_pico_s']
                if cerca.sum() < r['min_latidos']:
                    continue
                filas_ep.append({
                    'registro': registro, 'derivacion': k,
                    'nombre_derivacion': nombre, 'cadena': cadena,
                    'pico_s': ep['pico_s'],
                    'anotada_mv': ep['desviacion_max_mv'],
                    'medida_mv': np.median(valor[cerca]) - referencia,
                })

            fuera = np.ones(len(t_s), dtype=bool)
            for _, ep in eps.iterrows():
                fuera &= ~((t_s >= ep['inicio_s']) & (t_s <= ep['fin_s']))
            dif = np.diff(valor[fuera])
            mad = np.median(np.abs(dif - np.median(dif)))
            filas_ruido.append({'registro': registro, 'derivacion': k,
                                'nombre_derivacion': nombre, 'cadena': cadena,
                                'ruido_st_mv': 1.4826 * mad / np.sqrt(2)})
    return filas_ep, filas_ruido


def resumen_real(tabla_ep, tabla_ruido):
    """Por cadena: error absoluto medio, sesgo, correlación de Spearman y
    acierto del signo de la desviación por episodio, y mediana del ruido
    latido a latido. Devuelve un DataFrame (una fila por cadena)."""
    filas = []
    for cadena, g in tabla_ep.groupby('cadena', sort=False):
        error = g['medida_mv'] - g['anotada_mv']
        filas.append({
            'cadena': cadena,
            'n_episodios': len(g),
            'error_abs_medio_mv': error.abs().mean(),
            'sesgo_mv': error.mean(),
            'spearman': spearmanr(g['medida_mv'], g['anotada_mv']).statistic,
            'signo_correcto': (np.sign(g['medida_mv']) == np.sign(g['anotada_mv'])).mean(),
            'ruido_st_mediana_mv': tabla_ruido.loc[tabla_ruido['cadena'] == cadena,
                                                   'ruido_st_mv'].median(),
        })
    return pd.DataFrame(filas).set_index('cadena')


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def estilo_eje(eje):
    """Ejes recesivos: sin marco arriba y a la derecha, grilla suave."""
    eje.spines[['top', 'right']].set_visible(False)
    eje.grid(color='#e6e6e6', linewidth=0.8)
    eje.set_axisbelow(True)


def figura_respuestas(fs, impulsos, p, salida):
    """Respuesta en frecuencia (fase cero) de los filtros lineales y salida
    de los pasaaltos ante el impulso de 3 mV x 100 ms."""
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))

    f_baja = np.logspace(-3, np.log10(5), 400)
    for nombre, color in (('pa_005', COLOR_1), ('pa_067', COLOR_2)):
        ejes[0].semilogx(f_baja, respuesta_fase_cero(p['filtros'][nombre], fs, f_baja),
                         color=color, linewidth=2, label=nombre)
    ejes[0].set_xlabel('Frecuencia (Hz)')
    ejes[0].set_ylabel('|H(f)|² (ida y vuelta)')
    ejes[0].set_title('Pasaaltos', loc='left')
    ejes[0].legend(frameon=False)

    f_alta = np.linspace(0.5, fs / 2, 800)
    for nombre, color in (('pb_40', COLOR_1), ('notch_50', COLOR_2)):
        ejes[1].plot(f_alta, respuesta_fase_cero(p['filtros'][nombre], fs, f_alta),
                     color=color, linewidth=2, label=nombre)
    ejes[1].set_xlabel('Frecuencia (Hz)')
    ejes[1].set_title('Pasabajos y ranura', loc='left')
    ejes[1].legend(frameon=False)

    lim = p['impulso']['desplazamiento_max_mv']
    ejes[2].axhspan(-lim, lim, color='#e6e6e6', label=f'±{lim} mV (AHA/IEC)')
    for nombre, color in (('pa_005', COLOR_1), ('pa_067', COLOR_2)):
        t_s, y = impulsos[nombre]
        ejes[2].plot(t_s, y, color=color, linewidth=2, label=nombre)
    ejes[2].set_xlim(-1, 3)
    ejes[2].set_ylim(-0.4, 0.4)
    ejes[2].set_xlabel('Tiempo desde el impulso (s)')
    ejes[2].set_ylabel('Salida (mV)  [recortada]')
    ejes[2].set_title('Impulso 3 mV × 100 ms', loc='left')
    ejes[2].legend(frameon=False, loc='lower right')
    for eje in ejes:
        estilo_eje(eje)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def figura_ejemplo(base, latidos, ventanas, p, salida):
    """Un tramo de señal real con mucha deriva, cruda y después de cada
    cadena (un panel por cadena, mismo eje vertical). El tramo de
    config/filtrado.json debe estar fuera de ventanas descartadas: ahí las
    splines no tienen nodos e interpolan a través del hueco."""
    e = p['ejemplo']
    x_mv, fs, nombres = leer_senal_mv(base, e['registro'])
    x = x_mv[:, e['derivacion']]
    idx_qrs = latidos.loc[(latidos['registro'] == e['registro'])
                          & (latidos['derivacion'] == e['derivacion']), 'idx_qrs'].to_numpy()
    descartada = mascara_descarte(ventanas, e['registro'], e['derivacion'], len(x), fs)
    i0 = int(e['inicio_s'] * fs)
    i1 = i0 + int(e['duracion_s'] * fs)
    t_s = np.arange(i0, i1) / fs

    cadenas = list(p['cadenas'])
    fig, ejes = plt.subplots(len(cadenas), 1, figsize=(10, 1.5 * len(cadenas)),
                             sharex=True, sharey=True)
    for eje, cadena in zip(ejes, cadenas):
        y = aplicar_cadena(x, fs, idx_qrs, descartada, cadena, p)[i0:i1]
        eje.plot(t_s, y - np.median(y) if cadena == 'crudo' else y,
                 color=COLOR_1, linewidth=0.8)
        eje.axhline(0, color='#999999', linewidth=0.6)
        eje.set_ylabel('mV')
        eje.set_title(cadena + ('  (centrada en su mediana)' if cadena == 'crudo' else ''),
                      loc='left', fontsize=9)
        estilo_eje(eje)
    ejes[-1].set_xlabel(f'Tiempo (s) — {e["registro"]}, {nombres[e["derivacion"]]}')
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def grafico_puntos(eje, tabla, columna, cadenas, titulo):
    """Gráfico de puntos horizontal: una fila por cadena, un punto por
    condición (sin ruido en azul, con ruido en naranja)."""
    for k, (condicion, color) in enumerate((('sin_ruido', COLOR_1), ('con_ruido', COLOR_2))):
        valores = (tabla[tabla['condicion'] == condicion]
                   .groupby('cadena')[columna].mean().reindex(cadenas))
        eje.plot(valores, np.arange(len(cadenas)) + (k - 0.5) * 0.2, 'o',
                 color=color, markersize=8, label=condicion.replace('_', ' '))
    eje.set_yticks(np.arange(len(cadenas)), cadenas)
    eje.invert_yaxis()
    eje.set_xlabel(titulo)
    estilo_eje(eje)


def figura_sintetica(tabla, p, salida):
    """Error del ST latido a latido (RMSE) y error del cambio de ST durante
    el episodio, por cadena, sin y con ruido (promedio de plantillas y
    repeticiones)."""
    cadenas = list(p['cadenas'])
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    tabla = tabla.assign(error_cambio_abs_mv=tabla['error_cambio_mv'].abs())
    grafico_puntos(ejes[0], tabla, 'rmse_mv', cadenas, 'RMSE del ST latido a latido (mV)')
    grafico_puntos(ejes[1], tabla, 'error_cambio_abs_mv', cadenas,
                   f'|error| del cambio de ST en el episodio (mV)\n'
                   f'[verdadero: {p["sintetica"]["episodio"]["amplitud_mv"]} mV]')
    for eje in ejes:
        eje.set_xlim(left=0)
    ejes[1].legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def figura_real(tabla_ep, resumen, p, salida):
    """(a) desviación medida vs anotada con pa_005 y pa_067; (b) error
    absoluto medio y (c) ruido latido a latido, por cadena."""
    cadenas = list(p['cadenas'])
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4))
    for nombre, color in (('pa_005', COLOR_1), ('pa_067', COLOR_2)):
        g = tabla_ep[tabla_ep['cadena'] == nombre]
        ejes[0].plot(g['anotada_mv'], g['medida_mv'], 'o', color=color, markersize=4,
                     alpha=0.6, markeredgewidth=0, label=nombre)
    lim = [-1.0, 1.0]
    ejes[0].plot(lim, lim, color='#999999', linewidth=1)
    ejes[0].set_xlim(lim)
    ejes[0].set_ylim(lim)
    ejes[0].set_xlabel('Desviación anotada (mV)')
    ejes[0].set_ylabel('Desviación medida (mV)')
    ejes[0].legend(frameon=False)

    for eje, columna, titulo in ((ejes[1], 'error_abs_medio_mv',
                                  'Error absoluto medio por episodio (mV)'),
                                 (ejes[2], 'ruido_st_mediana_mv',
                                  'Ruido latido a latido del ST, mediana (mV)')):
        valores = resumen[columna].reindex(cadenas)
        eje.plot(valores, np.arange(len(cadenas)), 'o', color=COLOR_1, markersize=8)
        eje.set_yticks(np.arange(len(cadenas)), cadenas)
        eje.invert_yaxis()
        eje.set_xlabel(titulo)
    for eje in ejes:
        estilo_eje(eje)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    cfg = leer_config(base)
    p = leer_parametros()
    registros = sys.argv[2:] or (DATOS / cfg['carpeta'] / cfg['archivo_registros']).read_text().split()
    latidos = pd.read_csv(RESULTS / f'qrs_{base}.csv.gz')
    ventanas = pd.read_csv(RESULTS / f'calidad_ventanas_{base}.csv.gz')
    episodios = pd.read_csv(RESULTS / f'episodios_{base}.csv')
    _, fs, _ = leer_senal_mv(base, registros[0])
    carpeta_fig = RESULTS / 'figuras'
    carpeta_fig.mkdir(parents=True, exist_ok=True)

    # 1. Impulso.
    print('=== Prueba de impulso (AHA/IEC) ===')
    resultados, impulsos = [], {}
    for nombre in ('pa_005', 'pa_067'):
        res, impulsos[nombre] = prueba_impulso(nombre, fs, p)
        resultados.append(res)
    tabla_imp = pd.DataFrame(resultados)
    print(tabla_imp.to_string(index=False, float_format='%.3f'))
    figura_respuestas(fs, impulsos, p, carpeta_fig / f'filtrado_respuestas_{base}.png')

    # 2. Sintética.
    print('\n=== Señal sintética ===')
    tabla_sint = evaluar_sintetica(base, latidos, ventanas, episodios, fs, p)
    print(tabla_sint.groupby(['condicion', 'cadena'], sort=False)
          [['sesgo_mv', 'rmse_mv', 'error_cambio_mv']].mean()
          .to_string(float_format='%.4f'))
    figura_sintetica(tabla_sint, p, carpeta_fig / f'filtrado_sintetica_{base}.png')

    # 3. Real.
    print('\n=== Señal real ===')
    filas_ep, filas_ruido = [], []
    for registro in registros:
        ep, ru = evaluar_real(base, registro, latidos, ventanas, episodios, p)
        filas_ep.extend(ep)
        filas_ruido.extend(ru)
        print(f'  {registro}: {len(ep) // len(p["cadenas"])} episodios ST medidos')
    tabla_ep = pd.DataFrame(filas_ep)
    tabla_ruido = pd.DataFrame(filas_ruido)
    resumen = resumen_real(tabla_ep, tabla_ruido)
    print(resumen.to_string(float_format='%.3f'))
    figura_real(tabla_ep, resumen, p, carpeta_fig / f'filtrado_real_{base}.png')
    figura_ejemplo(base, latidos, ventanas, p, carpeta_fig / f'filtrado_ejemplo_{base}.png')

    tabla_imp.to_csv(RESULTS / f'filtrado_impulso_{base}.csv', index=False, float_format='%.4g')
    tabla_sint.to_csv(RESULTS / f'filtrado_sintetica_{base}.csv', index=False, float_format='%.4g')
    tabla_ep.to_csv(RESULTS / f'filtrado_real_episodios_{base}.csv', index=False, float_format='%.4g')
    tabla_ruido.to_csv(RESULTS / f'filtrado_real_ruido_{base}.csv', index=False, float_format='%.4g')
    print(f'\nGuardado en {RESULTS} y {carpeta_fig}')
