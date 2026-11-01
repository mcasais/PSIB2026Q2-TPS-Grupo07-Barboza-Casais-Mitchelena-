"""Serie temporal del nivel de ST latido a latido e interpolada (Paso 6).

Uso:
    python src/st_serie.py [base]
Salida:
    results/st_serie_<base>.csv.gz
"""
import json
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from episodios import leer_config
from exploracion import mascara_descarte
from qrs import leer_senal_mv
from rutas import CONFIG, RESULTS, DATOS

def leer_parametros():
    with open(CONFIG / 'st.json', encoding='utf-8') as f:
        return json.load(f)

def medir_st_latido(x_mv, fs, idx_qrs, p):
    n_samples = len(x_mv)
    
    #Convierto a ms y después descarto los latidos cuyos intervalos estén muy cerca del principio o del final de la señal y, al medir, queden cortados
    
    ini_iso = int(round(p['isoelectrico']['busqueda_ms'][0] / 1000 * fs))
    fin_iso = int(round(p['isoelectrico']['busqueda_ms'][1] / 1000 * fs))
    ancho_iso = int(round(p['isoelectrico']['ancho_ms'] / 1000 * fs))
    
    validos = (idx_qrs + ini_iso >= 0) & (idx_qrs + fin_iso <= n_samples)
    idx = idx_qrs[validos]
    
    tramos = x_mv[idx[:, None] + np.arange(ini_iso, fin_iso)] #saco la matriz con las ventanas que incluyen PR para cada latido.
    #Las tres siguientes líneas miden la variación interna de la señal con VENTANAS.
    var = np.abs(np.diff(tramos, axis=1))
    acum = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(var, axis=1)], axis=1)
    suma = acum[:, ancho_iso - 1:] - acum[:, :-(ancho_iso - 1)]
    #Estas tres líneas miden el valor medio de la señal en la ventana que tiene menor variación interna (la más plana) y lo toma como nivel isoelectrico. --> Encuentran la referencia.
    ini_min = np.argmin(suma, axis=1)
    cols = ini_min[:, None] + np.arange(ancho_iso)
    iso_mv = np.take_along_axis(tramos, cols, axis=1).mean(axis=1)
    
    #Cálculo de la duración de los intervalos RR y de la frecuencia cardíaca para cada latido. Se usa para decidir si el latido es taquicárdico o no, y así medir el ST en un punto más alejado del QRS.
    rr_s = np.diff(idx) / fs
    rr_s = np.append(rr_s, rr_s[-1] if len(rr_s) > 0 else 0.8)
    fc_bpm = 60.0 / np.maximum(rr_s, 0.3)
    
    #Después de tener la referencia, se mide el nivel de ST en cada latido, en un punto que depende de si el latido es taquicárdico o no. Se toma la media de la señal en una ventana centrada en ese punto y se resta el nivel isoelectrico.
    st_mv = np.zeros(len(idx))
    ancho_st = int(round(p['medicion_st']['ancho_ms'] / 1000 * fs)) // 2
    
    for i, (qrs_i, fc, iso) in enumerate(zip(idx, fc_bpm, iso_mv)):
        offset_ms = (p['medicion_st']['offset_taquicardia_ms'] 
                     if fc >= p['medicion_st']['umbral_taquicardia_bpm'] 
                     else p['medicion_st']['offset_normal_ms'])
        #EXPLICACIÓN DE PORQUE HICIMOS ESTO: Si la FC es => 120bpm (taquicardia), el punto J se acerca a la onda R y el ST se mide a 100ms (J+60ms).De lo contrario, se mide a 120ms (J+80ms). Por eso, el offset depende de la FC. El ancho de la ventana es fijo (40ms).
        #Hacemos esto porque a frecuencias cardíacas altas, el segmento ST se acorta y necesitamos medirlo más cerca del QRS para obtener una medición precisa, sino el punto J (donde termina el QRS) cae en la onda T y no en el segmento.
        centro = int(round(offset_ms / 1000 * fs))
        if qrs_i + centro + ancho_st < n_samples:
            tramo_st = x_mv[qrs_i + centro - ancho_st : qrs_i + centro + ancho_st + 1]
            st_mv[i] = tramo_st.mean() - iso
        else:
            st_mv[i] = np.nan
            
    return pd.DataFrame({'t_s': idx / fs, 'st_mv': st_mv, 'fc_bpm': fc_bpm})

def procesar_registro(base, registro, latidos, ventanas, p):
    cfg = leer_config(base)
    x_mv, fs, nombres = leer_senal_mv(base, registro)
    fs_grid = p['fs_remuestreo_hz']
    duracion_s = len(x_mv) / fs
    grid_t = np.arange(0, duracion_s, 1.0 / fs_grid)
    #Creamos un vector de tiempo uniforme con paso de $0,5s (2Hz) desde t=0 hasta el final del registro. Itera sobre cada señal y recupera la posición de los latidos.
    filas = []
    for k, nombre in enumerate(nombres):
        idx_qrs = latidos.loc[(latidos['registro'] == registro) & 
                              (latidos['derivacion'] == k), 'idx_qrs'].to_numpy()
        if len(idx_qrs) < 10:
            continue
        
        #Se descartan los latidos que no cumplen con la calidad de ventana mínima. Se obtiene un vector booleano donde True indica que el latido debe ser descartado (por ruido o mala calidad de señal).    
        descartada = mascara_descarte(ventanas, registro, k, len(x_mv), fs)
        df_st = medir_st_latido(x_mv[:, k], fs, idx_qrs, p)
        
        idx_validos = ~descartada[(df_st['t_s'] * fs).astype(int)]
        df_val = df_st[idx_validos].dropna()
        
        if len(df_val) < 10:
            continue
            
        st_suave = df_val['st_mv'].rolling(p['suavizado_latidos'], center=True, min_periods=1).median()   #Aplicación de filtro de mediana móvil para suavizar, lo cual ayuda a reducir el ruido y las fluctuaciones rápidas en la señal de ST, proporcionando una representación más estable y confiable del nivel de ST a lo largo del tiempo.
        f_interp = interp1d(df_val['t_s'], st_suave, kind='linear', bounds_error=False, fill_value=np.nan)
        st_grid = f_interp(grid_t)
        
        desc_grid = interp1d(np.arange(len(x_mv))/fs, descartada.astype(float), kind='nearest', bounds_error=False)(grid_t)
        st_grid[desc_grid > 0.5] = np.nan  #Se vuelve a determinar NaN (no válido o nulo) a los tramos que correspondan a ventanas descartadas por ruido, así no inventamos datos en huecos.
        
        #Creamos una tabla con los datos obtenidos unificados.
        temp = pd.DataFrame({
            'registro': registro,
            'derivacion': k,
            'nombre_derivacion': nombre,
            't_s': grid_t,
            'st_mv': st_grid
        })
        filas.append(temp)
        
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()

#Lo que hace esta función es procesar todos los registros de la base de datos, medir el nivel de ST latido a latido, suavizarlo y remuestrearlo a una frecuencia uniforme. Luego guarda los resultados en un archivo CSV comprimido.
if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    cfg = leer_config(base)
    p = leer_parametros()
    registros = (DATOS / cfg['carpeta'] / cfg['archivo_registros']).read_text().split()
    
    latidos = pd.read_csv(RESULTS / f'qrs_{base}.csv.gz')
    ventanas = pd.read_csv(RESULTS / f'calidad_ventanas_{base}.csv.gz')
    
    res = []
    for r in registros:
        df_r = procesar_registro(base, r, latidos, ventanas, p)
        if not df_r.empty:
            res.append(df_r)
            
    df_final = pd.concat(res, ignore_index=True)
    salida = RESULTS / f'st_serie_{base}.csv.gz'
    df_final.to_csv(salida, index=False, float_format='%.4g')
    print(f'Paso 6 completado. Guardado en {salida}')
