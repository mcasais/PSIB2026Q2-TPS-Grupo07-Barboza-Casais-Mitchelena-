"""Segmentación por coeficiente de disimilitud (Paso 7).

Uso:
    python src/segmentacion.py [base]
Salida:
    results/segmentacion_<base>.csv
"""

#**EXPLICACIÓN:** 
#El objetivo de este script es detectar automáticamente episodios de cambio en el segmento ST analizando la serie temporal continua generada en el Paso 6 (detección serie ST). Para ello compara una ventana "de referencia" (300s previos) contra una ventana "de prueba" (60s posteriores) mediante un coeficiente de disimilitud eta.

import json
import sys
import numpy as np
import pandas as pd

from episodios import leer_config
from rutas import CONFIG, RESULTS

def leer_parametros():
    with open(CONFIG / 'segmentacion.json', encoding='utf-8') as f:
        return json.load(f)

def calcular_disimilitud(st_serie, p): #Creamos una función que calcula el coeficiente de disimilitud eta entre dos ventanas de la serie temporal de ST. La ventana de referencia es de 300s y la ventana de prueba es de 60s (como la señal está muestreada a 2Hz, 300s de referencia equivalen a 600 muestras y 60s de prueba equivalen a 120 muestras). Se calcula la media y desviación estándar de la ventana de referencia y se compara con la media de la ventana de prueba. El resultado es un vector con los valores del coeficiente de disimilitud para cada punto en el tiempo.
    n_ref = int(p['ventana_ref_s'] * 2.0)
    n_test = int(p['ventana_test_s'] * 2.0)
    
    st_vals = st_serie.to_numpy()
    N = len(st_vals)
    d_coef = np.zeros(N)
    
    for i in range(n_ref, N - n_test): #Para cada punto i en el tiempo, se toma la ventana de referencia y la ventana de prueba, se eliminan los valores NaN (que representan datos no válidos o nulos) y se calcula la media y desviación estándar de la ventana de referencia. Luego se calcula la media de la ventana de prueba y se calcula el coeficiente de disimilitud eta como el valor absoluto de la diferencia entre las medias dividido por la desviación estándar de la ventana de referencia más un pequeño valor para evitar división por cero. Si alguna de las ventanas tiene menos de la mitad de sus valores válidos, se asigna un valor de 0 al coeficiente de disimilitud para ese punto.
        w_ref = st_vals[i - n_ref : i]
        w_test = st_vals[i : i + n_test]
        
        w_ref_val = w_ref[~np.isnan(w_ref)]
        w_test_val = w_test[~np.isnan(w_test)]
        
        if len(w_ref_val) < n_ref // 2 or len(w_test_val) < n_test // 2:
            d_coef[i] = 0.0
            continue
            
        m_ref, s_ref = np.mean(w_ref_val), np.std(w_ref_val)
        m_test = np.mean(w_test_val)
        
        d_coef[i] = np.abs(m_test - m_ref) / (s_ref + 1e-4)
        
    return d_coef

def detectar_segmentos(df_deriv, p): #Convierte al vector eta en un vector booleano que indica si el coeficiente de disimilitud supera un umbral definido en los parámetros. Luego encuentra los inicios y fines de los episodios donde el coeficiente supera el umbral y calcula la duración de cada episodio. Si la duración es mayor o igual a un valor mínimo definido en los parámetros, se guarda la información del episodio detectado (registro, derivación, inicio, fin y duración) en una lista de diccionarios. Finalmente, devuelve un DataFrame con los episodios detectados.
    t_s = df_deriv['t_s'].to_numpy()
    d_coef = calcular_disimilitud(df_deriv['st_mv'], p)
    
    sobre_umbral = d_coef >= p['umbral_disimilitud_eta']
    bordes = np.diff(np.r_[0, sobre_umbral.astype(int), 0])
    inicios = np.flatnonzero(bordes == 1)
    fines = np.flatnonzero(bordes == -1)
    
    episodios_det = []
    #Solo conserva como episodios válidos las detecciones que duren al menos 30s (descartando fluctuaciones breves o artefactos).
    for i, f in zip(inicios, fines):
        dur = t_s[f - 1] - t_s[i]
        if dur >= p['duracion_min_episodio_s']:
            episodios_det.append({
                'registro': df_deriv['registro'].iloc[0],
                'derivacion': df_deriv['derivacion'].iloc[0],
                'inicio_det_s': t_s[i],
                'fin_det_s': t_s[f - 1],
                'duracion_det_s': dur
            })
    return pd.DataFrame(episodios_det)

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    p = leer_parametros()
    
    st_df = pd.read_csv(RESULTS / f'st_serie_{base}.csv.gz')
    gt_df = pd.read_csv(RESULTS / f'episodios_{base}.csv')
    
    detecciones = []
    for (r, d), grupo in st_df.groupby(['registro', 'derivacion']):
        det = detectar_segmentos(grupo, p)
        if not det.empty:
            detecciones.append(det)
            
    df_det = pd.concat(detecciones, ignore_index=True) if detecciones else pd.DataFrame()
    
    salida = RESULTS / f'segmentacion_{base}.csv'
    df_det.to_csv(salida, index=False)
    print(f'Paso 7 completado. Detecciones guardadas en {salida}')
