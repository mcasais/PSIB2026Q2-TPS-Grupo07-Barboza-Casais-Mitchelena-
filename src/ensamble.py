"""Ensamble del perfil de ST alineado al inicio del episodio (Paso 8).

Uso:
    python src/ensamble.py [base]
Salida:
    results/ensamble_<base>.csv
    results/figuras/ensamble_st_<base>.png
"""

#**EXPLICACIÓN:**
# Este script construye un perfil promedio (ensamble) para observar cómo evoluciona el nivel del segmento ST antes, durante y después de un episodio isquémico real. 
# Para comparar episodios de distintos pacientes y duraciones, se alinean todos los eventos anotados en t=0s (el inicio del episodio) y resta a cada uno su nivel de ST basal previo.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rutas import RESULTS

COLOR_1 = '#2a78d6'

def construir_ensamble(st_df, gt_df, t_pre=300, t_post=900, fs=2.0): #Definimos una ventana de tiempo relativa al inicio del episodio (t=0s) que va desde 300s antes hasta 900s después, con un muestreo de 2Hz (cada 0.5s).
    grid_rel = np.arange(-t_pre, t_post, 1.0 / fs)
    matriz_eps = [] #Lista donde se guardará la curva de ST alineada de cada episodio individual.
    
    gt_st = gt_df[(gt_df['tipo'] == 'ST') & (~gt_df['cambio_eje'])] #Filtramos los episodios anotados de tipo ST y descartamos aquellos que tengan un cambio de eje, ya que podrían representar artefactos o eventos no representativos.
    
    #Para cada episodio anotado en el segundo t_ini, calcula los tiempos absolutos (t_abs) en el registro y extrae/interpola los valores de ST que ocurrieron en ese rango relativo de [-300s, +900s].
    for _, ep in gt_st.iterrows():
        reg, deriv, t_ini = ep['registro'], ep['derivacion'], ep['inicio_s']
        
        sub = st_df[(st_df['registro'] == reg) & (st_df['derivacion'] == deriv)]
        if sub.empty:
            continue
            
        t_abs = t_ini + grid_rel
        st_interp = np.interp(t_abs, sub['t_s'], sub['st_mv'], left=np.nan, right=np.nan)
        
        idx_basal = (grid_rel >= -300) & (grid_rel <= -60) #Calcula el ST promedio del paciente durante la ventana reposada de -300s a -60s previos al episodio (basal) y se lo resta a toda la curva (st_interp - basal).
        basal = np.nanmedian(st_interp[idx_basal]) #Esto se hace porque cada paciente tiene una línea de base de ST distinta. Al restar el valor basal, todas las curvas arrancan desde 0mV y miden únicamente la variación o desviación en mV causada por la isquemia.
        
        if not np.isnan(basal):
            matriz_eps.append(st_interp - basal)
            
    matriz = np.array(matriz_eps)
    
    # Para cada instante de tiempo relativo, calcula la mediana y los percentiles 25 y 75 de todos los episodios alineados. Esto nos da una curva promedio (mediana) y un rango intercuartílico (RIC) que representa la variabilidad entre pacientes.
    mediana = np.nanmedian(matriz, axis=0)
    q25 = np.nanpercentile(matriz, 25, axis=0)
    q75 = np.nanpercentile(matriz, 75, axis=0)
    
    return pd.DataFrame({
        't_relativo_s': grid_rel,
        'st_mediana_mv': mediana,
        'st_q25_mv': q25,
        'st_q75_mv': q75
    })

# Guarda el ensamble en un CSV y genera una figura con la mediana y el rango intercuartílico (RIC) de la serie temporal de ST alineada al inicio del episodio. La figura muestra cómo varía el nivel de ST antes, durante y después del evento isquémico, permitiendo visualizar patrones comunes entre pacientes.
if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    
    st_df = pd.read_csv(RESULTS / f'st_serie_{base}.csv.gz')
    gt_df = pd.read_csv(RESULTS / f'episodios_{base}.csv')
    
    ensamble = construir_ensamble(st_df, gt_df)
    
    salida_csv = RESULTS / f'ensamble_{base}.csv'
    ensamble.to_csv(salida_csv, index=False)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ensamble['t_relativo_s'], ensamble['st_mediana_mv'], color=COLOR_1, lw=2, label='Mediana')
    ax.fill_between(ensamble['t_relativo_s'], ensamble['st_q25_mv'], ensamble['st_q75_mv'],
                    color=COLOR_1, alpha=0.25, label='RIC (p25–p75)')
    ax.axvline(0, color='k', linestyle='--', alpha=0.7, label='Inicio anotado')
    ax.set_xlabel('Tiempo relativo al inicio (s)')
    ax.set_ylabel('Cambio de ST (mV)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#e6e6e6')
    ax.legend(frameon=False)
    fig.tight_layout()
    
    salida_fig = RESULTS / 'figuras' / f'ensamble_st_{base}.png'
    fig.savefig(salida_fig, dpi=150)
    plt.close(fig)
    print(f'Paso 8 completado. Guardado en {salida_csv} y {salida_fig}')
