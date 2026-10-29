"""Detección de QRS con Pan-Tompkins (paso 5, adelantado para el filtrado).

Etapas (Pan y Tompkins 1985, IEEE Trans Biomed Eng 32:230):
    1. Pasabanda: deja la energía del QRS y atenúa P, T, deriva y ruido
       muscular. El original usa 5-15 Hz; acá se usa 8-20 Hz (ver abajo).
    2. Derivada de 5 puntos: resalta las pendientes empinadas del QRS.
    3. Cuadrado: todo positivo y realza las pendientes grandes.
    4. Integración en ventana móvil de 150 ms (ancho de un QRS ancho):
       convierte cada QRS en un solo lóbulo.
    5. Umbrales adaptativos sobre la señal integrada, con período
       refractario, discriminación de onda T y búsqueda hacia atrás.

Diferencias con el original, elegidas porque el análisis es fuera de línea:
    - Filtros de fase cero (sosfiltfilt) y ventana de integración centrada:
      no hay retardo que compensar.
    - Los coeficientes del original son enteros para fs = 200 Hz; acá se
      diseña un Butterworth para la fs del registro (la EDB está a 250 Hz).
    - Pasabanda 8-20 Hz en lugar de 5-15 Hz: en derivaciones con ondas T
      grandes (p. ej. V2 de e0305) la T pasaba el umbral y su pendiente
      superaba la mitad de la del QRS, así que se detectaba como un segundo
      latido. La T tiene menos contenido por encima de 8 Hz que el QRS.
      Sobre toda la EDB (sin ventanas descartadas): VPP 0,982 -> 0,995 y
      Se 0,9984 -> 0,9987; derivaciones con VPP < 0,95: 11 -> 2.
    - Reinicio de umbrales tras `reinicio_sin_qrs_s` sin detecciones (ver
      umbralizar).
    - Se usa solo el juego de umbrales de la señal integrada (el original
      tiene además uno sobre la señal filtrada) y no se reducen los umbrales
      a la mitad con RR irregular. Es la versión de los apuntes del curso.

El detector se aplica a cada derivación por separado: el filtrado de cada
derivación (src/filtrado.py) usa sus propios latidos.

Uso (desde la raíz del repo):
    python src/qrs.py [base]      # base por defecto: edb
Salidas:
    results/qrs_<base>.csv.gz            un latido detectado por fila
    results/validacion_qrs_<base>.csv    Se y VPP por registro y derivación
"""
import json
import sys

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, find_peaks, sosfiltfilt

from episodios import leer_config
from exploracion import SIMBOLOS_LATIDO, mascara_descarte
from rutas import CONFIG, DATOS, RESULTS


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/qrs.json."""
    with open(CONFIG / 'qrs.json', encoding='utf-8') as f:
        return json.load(f)


def leer_senal_mv(base, registro):
    """Lee un registro y devuelve (x_mv, fs, nombres): x_mv es una matriz
    (muestras x derivaciones) en mV y fs la frecuencia de muestreo en Hz."""
    cfg = leer_config(base)
    rec = wfdb.rdrecord(str(DATOS / cfg['carpeta'] / registro))
    return rec.p_signal, rec.fs, rec.sig_name


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def preprocesar(x_mv, fs, p):
    """Etapas 1 a 4 de Pan-Tompkins.

    Entradas: x_mv (señal de una derivación, mV), fs (Hz), p (parámetros).
    Salidas: x_pb (pasabanda, mV), derivada (mV/s) e integrada (mV²/s²,
    solo importa su forma relativa).
    """
    sos = butter(p['orden_pasabanda'], p['pasabanda_hz'], btype='bandpass',
                 fs=fs, output='sos')
    x_pb = sosfiltfilt(sos, x_mv)

    # Derivada de 5 puntos centrada: y[n] = (−x[n−2] − 2x[n−1] + 2x[n+1] +
    # x[n+2]) · fs/8. np.convolve invierte el núcleo, por eso va al revés.
    nucleo = np.array([1, 2, 0, -2, -1]) * fs / 8
    derivada = np.convolve(x_pb, nucleo, mode='same')

    n_int = int(round(p['ventana_integracion_s'] * fs))
    integrada = np.convolve(derivada ** 2, np.ones(n_int) / n_int, mode='same')
    return x_pb, derivada, integrada


def umbralizar(integrada, derivada, fs, p):
    """Etapa 5: decide qué picos de la señal integrada son QRS.

    Mantiene dos niveles que se actualizan con un promedio exponencial:
    SPKI (nivel de los picos de QRS) y NPKI (nivel de los picos de ruido).
    El umbral es THR1 = NPKI + 0,25·(SPKI − NPKI).
    - Período refractario: dos QRS no pueden estar a menos de 200 ms
      (se impone al buscar los picos).
    - Onda T: un pico a menos de 360 ms del QRS anterior cuya pendiente
      máxima es menor que la mitad de la de ese QRS se toma como onda T.
    - Búsqueda hacia atrás: si pasan más de 1,66 RR promedio sin QRS, se
      acepta el mayor pico de ruido de ese intervalo si supera THR1/2.

    Entradas: señal integrada, derivada (de preprocesar), fs (Hz), p.
    Salida: vector de índices (muestras) de los picos de la integrada
    aceptados como QRS.
    """
    n_int = int(round(p['ventana_integracion_s'] * fs))
    n_refractario = int(round(p['refractario_s'] * fs))
    n_onda_t = int(round(p['ventana_onda_t_s'] * fs))
    picos, _ = find_peaks(integrada, distance=n_refractario)

    # Inicialización con los primeros segundos (fase de aprendizaje).
    n_apr = int(round(p['aprendizaje_s'] * fs))
    spki = 0.25 * integrada[:n_apr].max()
    npki = 0.5 * integrada[:n_apr].mean()

    def pendiente_max(idx):
        # Pendiente máxima en la ventana de integración que termina en idx.
        return np.abs(derivada[max(idx - n_int, 0):idx + 1]).max()

    qrs = []                 # índices aceptados como QRS
    rr = []                  # intervalos RR (muestras)
    pendiente_ultimo = None  # pendiente máxima del último QRS
    ruido_pendiente = []     # picos de ruido desde el último QRS

    def aceptar(idx, peso):
        nonlocal spki, pendiente_ultimo
        if qrs:
            rr.append(idx - qrs[-1])
        qrs.append(idx)
        spki = peso * integrada[idx] + (1 - peso) * spki
        pendiente_ultimo = pendiente_max(idx)

    n_reinicio = int(round(p['reinicio_sin_qrs_s'] * fs))

    for pk in picos:
        # Reinicio (agregado al original): si un artefacto grande infló SPKI,
        # el umbral queda por encima de todos los QRS siguientes y el
        # detector no se recupera nunca (pasaba en e0611). Si pasan más de
        # `reinicio_sin_qrs_s` sin QRS, se vuelve a aprender con los
        # últimos segundos, como al principio del registro.
        if qrs and pk - qrs[-1] > n_reinicio:
            tramo = integrada[max(pk - n_apr, 0):pk + 1]
            spki = 0.25 * tramo.max()
            npki = 0.5 * tramo.mean()
            ruido_pendiente = []

        umbral_1 = npki + 0.25 * (spki - npki)

        # Búsqueda hacia atrás: ¿se perdió un latido desde el último QRS?
        if rr and ruido_pendiente:
            rr_prom = np.mean(rr[-p['n_rr_promedio']:])
            if pk - qrs[-1] > p['factor_busqueda_atras'] * rr_prom:
                candidato = max(ruido_pendiente, key=lambda i: integrada[i])
                if integrada[candidato] > 0.5 * umbral_1:
                    # En la búsqueda hacia atrás el original pesa 0,25.
                    aceptar(candidato, peso=0.25)
                    ruido_pendiente = [i for i in ruido_pendiente if i > candidato]
                    umbral_1 = npki + 0.25 * (spki - npki)

        es_qrs = integrada[pk] > umbral_1
        if es_qrs and qrs and pk - qrs[-1] < n_onda_t:
            if pendiente_max(pk) < 0.5 * pendiente_ultimo:
                es_qrs = False          # onda T

        if es_qrs:
            aceptar(pk, peso=0.125)
            ruido_pendiente = []
        else:
            npki = 0.125 * integrada[pk] + 0.875 * npki
            ruido_pendiente.append(pk)

    return np.array(qrs, dtype=int)


def detectar_qrs(x_mv, fs, p):
    """Detecta los QRS de una derivación.

    Entradas: x_mv (señal cruda de una derivación, mV), fs (Hz), p
    (parámetros de config/qrs.json).
    Salida: idx_qrs, vector de índices (muestras) del pico del QRS,
    ubicado en el máximo de |señal pasabanda| cerca del pico de la
    integrada (±mitad de la ventana de integración, que es centrada).
    """
    x_pb, derivada, integrada = preprocesar(x_mv, fs, p)
    idx_integrada = umbralizar(integrada, derivada, fs, p)

    medio = int(round(p['ventana_integracion_s'] * fs / 2))
    idx_qrs = np.empty_like(idx_integrada)
    for k, idx in enumerate(idx_integrada):
        ini = max(idx - medio, 0)
        idx_qrs[k] = ini + np.argmax(np.abs(x_pb[ini:idx + medio + 1]))
    return idx_qrs


# ---------------------------------------------------------------------------
# Validación contra las anotaciones
# ---------------------------------------------------------------------------

def emparejar(idx_ref, idx_det, n_tol):
    """Empareja latidos de referencia y detectados, uno a uno.

    Recorre las dos listas ordenadas a la vez: si el detectado está a
    <= n_tol muestras de la referencia es un verdadero positivo (VP); si
    quedó antes, es un falso positivo (FP); si quedó después, la
    referencia no se detectó (falso negativo, FN).
    Entradas: idx_ref, idx_det (vectores ordenados, muestras), n_tol
    (muestras). Salida: (VP, FP, FN).
    """
    i, j = 0, 0
    vp = fp = fn = 0
    while i < len(idx_ref) and j < len(idx_det):
        if abs(idx_det[j] - idx_ref[i]) <= n_tol:
            vp += 1
            i += 1
            j += 1
        elif idx_det[j] < idx_ref[i]:
            fp += 1
            j += 1
        else:
            fn += 1
            i += 1
    fp += len(idx_det) - j
    fn += len(idx_ref) - i
    return vp, fp, fn


def procesar_registro(base, registro, ventanas, p):
    """Detecta y valida los QRS de las dos derivaciones de un registro.

    La validación se hace dos veces: con todo el registro y sin las
    ventanas descartadas por calidad (paso 3). En cada caso se sacan tanto
    las referencias como las detecciones que caen en esas ventanas.
    Salidas: (DataFrame de latidos detectados, lista de dicts de validación).
    """
    cfg = leer_config(base)
    x_mv, fs, nombres = leer_senal_mv(base, registro)
    anot = wfdb.rdann(str(DATOS / cfg['carpeta'] / registro),
                      cfg['extension_anotaciones'])
    idx_ref = np.array([s for s, sim in zip(anot.sample, anot.symbol)
                        if sim in SIMBOLOS_LATIDO])
    n_tol = int(round(p['tolerancia_validacion_s'] * fs))

    latidos, validacion = [], []
    for k, nombre in enumerate(nombres):
        idx_qrs = detectar_qrs(x_mv[:, k], fs, p)
        latidos.append(pd.DataFrame({'registro': registro, 'derivacion': k,
                                     'idx_qrs': idx_qrs}))

        descartada = mascara_descarte(ventanas, registro, k, len(x_mv), fs)
        for alcance, ref, det in (
                ('todo', idx_ref, idx_qrs),
                ('sin_descartadas', idx_ref[~descartada[idx_ref]],
                 idx_qrs[~descartada[idx_qrs]])):
            vp, fp, fn = emparejar(ref, det, n_tol)
            validacion.append({'registro': registro, 'derivacion': k,
                               'nombre_derivacion': nombre, 'alcance': alcance,
                               'vp': vp, 'fp': fp, 'fn': fn,
                               'sensibilidad': vp / (vp + fn) if vp + fn else np.nan,
                               'vpp': vp / (vp + fp) if vp + fp else np.nan})
    return pd.concat(latidos, ignore_index=True), validacion


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    cfg = leer_config(base)
    p = leer_parametros()
    registros = (DATOS / cfg['carpeta'] / cfg['archivo_registros']).read_text().split()
    ventanas = pd.read_csv(RESULTS / f'calidad_ventanas_{base}.csv.gz')

    latidos, validacion = [], []
    for registro in registros:
        lat, val = procesar_registro(base, registro, ventanas, p)
        latidos.append(lat)
        validacion.extend(val)
        print(f'  {registro}: ' + '  '.join(
            f'd{v["derivacion"]} Se={v["sensibilidad"]:.3f} VPP={v["vpp"]:.3f}'
            for v in val if v['alcance'] == 'todo'))
    latidos = pd.concat(latidos, ignore_index=True)
    validacion = pd.DataFrame(validacion)

    # Resumen: totales agregados (suma de VP, FP y FN de todos los registros,
    # como en la norma ANSI/AAMI EC57) y la distribución por derivación.
    print('\n=== Validación Pan-Tompkins (tolerancia '
          f'{p["tolerancia_validacion_s"] * 1000:.0f} ms) ===')
    for alcance, grupo in validacion.groupby('alcance'):
        vp, fp, fn = grupo[['vp', 'fp', 'fn']].sum()
        print(f'{alcance:16s} Se={vp / (vp + fn):.4f}  VPP={vp / (vp + fp):.4f}  '
              f'(VP={vp}, FP={fp}, FN={fn});  Se por derivación: mediana '
              f'{grupo["sensibilidad"].median():.4f}, mínima '
              f'{grupo["sensibilidad"].min():.3f};  VPP mínimo '
              f'{grupo["vpp"].min():.3f}')
    peores = (validacion[validacion['alcance'] == 'sin_descartadas']
              .assign(f1=lambda d: 2 * d['vp'] / (2 * d['vp'] + d['fp'] + d['fn']))
              .nsmallest(5, 'f1'))
    print('\nDerivaciones con peor desempeño (sin descartadas):')
    print(peores[['registro', 'nombre_derivacion', 'sensibilidad', 'vpp']]
          .to_string(index=False, float_format='%.3f'))

    salida_lat = RESULTS / f'qrs_{base}.csv.gz'
    latidos.to_csv(salida_lat, index=False)
    salida_val = RESULTS / f'validacion_qrs_{base}.csv'
    validacion.to_csv(salida_val, index=False, float_format='%.4f')
    print(f'\nGuardado en {salida_lat} y {salida_val}')
