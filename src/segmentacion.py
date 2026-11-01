"""Segmentación por coeficiente de disimilitud (Paso 7).

Uso:
    python src/segmentacion.py [base]
Salida:
    results/segmentacion_<base>.csv
"""
import json
import sys
import numpy as np
import pandas as pd

from episodios import leer_config
from rutas import CONFIG, RESULTS

def leer_parametros():
    with open(CONFIG / 'segmentacion.json', encoding='utf-8') as f:
        return json.load(f)

def calcular_disimilitud(st_serie, p):
    n_ref = int(p['ventana_ref_s'] * 2.0)
    n_test = int(p['ventana_test_s'] * 2.0)
    
    st_vals = st_serie.to_numpy()
    N = len(st_vals)
    d_coef = np.zeros(N)
    
    for i in range(n_ref, N - n_test):
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

def detectar_segmentos(df_deriv, p):
    t_s = df_deriv['t_s'].to_numpy()
    d_coef = calcular_disimilitud(df_deriv['st_mv'], p)
    
    sobre_umbral = d_coef >= p['umbral_disimilitud_eta']
    bordes = np.diff(np.r_[0, sobre_umbral.astype(int), 0])
    inicios = np.flatnonzero(bordes == 1)
    fines = np.flatnonzero(bordes == -1)
    
    episodios_det = []
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
