"""Análisis de VFC mediante periodograma de Lomb-Scargle (Paso 9).

Uso:
    python src/vfc.py [base]
Salida:
    results/vfc_<base>.csv
"""
#**EXPLICACIÓN:**
# Este script analiza los cambios del sistema simpático y parasimpático mediante el análisis espectral de la serie de intervalos RR. Como los latidos no ocurren a tiempos perfectamente regulares, se utiliza el periodograma de Lomb-Scargle en lugar de una FFT común (nos evitamos interpolar para generar equidistancia).

import json
import sys
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from episodios import leer_config
from rutas import CONFIG, RESULTS

def leer_parametros():
    with open(CONFIG / 'vfc.json', encoding='utf-8') as f:
        return json.load(f)

def calcular_bandas_lomb(t_rr, rr_s, p):
    if len(t_rr) < 15:
        return {'vlf': np.nan, 'lf': np.nan, 'hf': np.nan, 'lf_hf': np.nan}
        
    frecs = np.linspace(0.0033, 0.40, 500) #Definimos 500 puntos a evaluar entre 0.0033Hz y 0.4Hz, que es el rango de frecuencias de interés para el análisis de VFC.
    angular = 2 * np.pi * frecs
    
    pgram = lombscargle(t_rr, rr_s - np.mean(rr_s), angular, precenter=False) #Calcula la densidad de potencia espectral sobre la serie de tiempos de ocurrencia del RR (t_rr) sin necesidad de reinterpolar la señal.
    
   
    #Integración de las bandas: las bandas de frecuencia se definen en Hz y se integran sobre el periodograma para obtener la potencia en cada banda. Puedo obtener VLF (regulación hormonal), LF (regulación simpática) y HF (regulación parasimpática). Luego calculo la relación LF/HF, que es un indicador del balance simpático-parasimpático (HRV).
    b = p['bandas_hz']
    idx_vlf = (frecs >= b['vlf'][0]) & (frecs < b['vlf'][1])
    idx_lf = (frecs >= b['lf'][0]) & (frecs < b['lf'][1])
    idx_hf = (frecs >= b['hf'][0]) & (frecs <= b['hf'][1])
    
    df_f = frecs[1] - frecs[0]
    p_vlf = np.sum(pgram[idx_vlf]) * df_f
    p_lf = np.sum(pgram[idx_lf]) * df_f
    p_hf = np.sum(pgram[idx_hf]) * df_f
    
    return {
        'vlf': p_vlf,
        'lf': p_lf,
        'hf': p_hf,
        'lf_hf': p_lf / (p_hf + 1e-8)
    }

def procesar_vfc_episodio(ep, qrs_df, p, fs=250.0): #Extraemos los intervalos RR del registro y descartamos latidos ectópicos o fisiológicamente imposibles (RR<0,4s o RR>1,8s). 
    reg, deriv = ep['registro'], ep['derivacion']
    sub_qrs = qrs_df[(qrs_df['registro'] == reg) & (qrs_df['derivacion'] == deriv)]
    
    if len(sub_qrs) < 20:
        return None
        
    t_qrs = sub_qrs['idx_qrs'].to_numpy() / fs
    rr = np.diff(t_qrs)
    t_rr = t_qrs[1:]
    
    validos = (rr >= p['rr_min_s']) & (rr <= p['rr_max_s'])
    t_rr, rr = t_rr[validos], rr[validos]
    
    res_ep = {'registro': reg, 'derivacion': deriv, 'inicio_s': ep['inicio_s']}
    
    #Hacemos una división Temporal para evaluar los parámetros espectrales en tres momentos clave del episodio isquémico (antes, durante y después del evento) y así poder comparar estadísticamente cómo varía el tono simpático- parasimpático al desencadenarse el episodio ST.
    ventanas = {
        'antes': (ep['inicio_s'] + p['ventanas_s']['antes'][0], ep['inicio_s'] + p['ventanas_s']['antes'][1]),
        'durante': (ep['inicio_s'] + p['ventanas_s']['durante'][0], ep['inicio_s'] + p['ventanas_s']['durante'][1]),
        'despues': (ep['fin_s'] + p['ventanas_s']['despues'][0], ep['fin_s'] + p['ventanas_s']['despues'][1])
    }
    
    for nombre_v, (t0, t1) in ventanas.items():
        m = (t_rr >= t0) & (t_rr <= t1)
        bandas = calcular_bandas_lomb(t_rr[m], rr[m], p)
        for banda, val in bandas.items():
            res_ep[f'{banda}_{nombre_v}'] = val
            
    return res_ep

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    p = leer_parametros()
    
    qrs_df = pd.read_csv(RESULTS / f'qrs_{base}.csv.gz')
    gt_df = pd.read_csv(RESULTS / f'episodios_{base}.csv')
    gt_st = gt_df[(gt_df['tipo'] == 'ST') & (~gt_df['cambio_eje'])]
    
    filas = []
    for _, ep in gt_st.iterrows():
        res = procesar_vfc_episodio(ep, qrs_df, p)
        if res:
            filas.append(res)
            
    df_vfc = pd.DataFrame(filas)
    salida = RESULTS / f'vfc_{base}.csv'
    df_vfc.to_csv(salida, index=False, float_format='%.4g')
    print(f'Paso 9 completado. Resultados guardados en {salida}') #Guardamos los resultados de la VFC en un archivo CSV comprimido para su posterior análisis y visualización.
