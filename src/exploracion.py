"""Exploración y acondicionamiento de la base (paso 3).

Tres partes:
    1. Derivaciones y datos clínicos de cada registro (del .hea).
    2. Estadística de los episodios anotados (de results/episodios_<base>.csv,
       generado por src/episodios.py).
    3. Calidad de señal por ventanas de duración fija, en cada derivación:
       - deriva de línea de base: excursión pico a pico de la línea de base
         estimada con doble filtro de mediana (200 ms y 600 ms, de Chazal
         2004), en mV;
       - ruido de alta frecuencia (muscular): desvío robusto (1,4826·MAD) de
         la señal filtrada con un pasaaltos de 40 Hz, en mV;
       - saturación: muestras en mesetas pegadas al máximo o mínimo del canal;
       - ruido anotado: fracción de la ventana marcada como ruidosa o
         ilegible por las anotaciones '~' de la base.
   Las métricas propias se guardan como números continuos, sin umbral. Para
   descartar ventanas se usa solo la anotación de la base (columna
   'descartar', ver marcar_descarte).

Uso (desde la raíz del repo):
    python src/exploracion.py [base]      # base por defecto: edb
Salidas:
    results/registros_<base>.csv            una fila por registro
    results/calidad_ventanas_<base>.csv.gz  una fila por ventana y derivación
    results/figuras/exploracion_*.png
"""
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')                 # solo se guardan figuras, sin ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu

from episodios import leer_config
from rutas import CONFIG, DATOS, RESULTS

# Máscara de bits del campo `subtype` de las anotaciones '~' (convención
# WFDB, compartida por la EDB y la LTST). Cada '~' fija el estado de calidad
# desde esa muestra hasta la próxima '~'.
#   bit 0 / bit 1 -> señal 0 / señal 1 ruidosa
#   bit 4 / bit 5 -> señal 0 / señal 1 ilegible
BIT_RUIDOSA = {0: 0x01, 1: 0x02}
BIT_ILEGIBLE = {0: 0x10, 1: 0x20}

# Símbolos de anotación que corresponden a latidos (convención WFDB).
SIMBOLOS_LATIDO = set('NLRBAaJSVrFejnE/fQ?')

# Colores (paleta categórica validada: azul y naranja).
COLOR_1 = '#2a78d6'
COLOR_2 = '#eb6834'


def leer_parametros():
    """Devuelve el diccionario de parámetros de config/calidad.json."""
    with open(CONFIG / 'calidad.json', encoding='utf-8') as f:
        return json.load(f)


def muestras_impar(t_s, fs):
    """Convierte una duración en segundos a un número impar de muestras
    (los filtros de mediana necesitan una ventana centrada)."""
    n = int(round(t_s * fs))
    return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# 1. Derivaciones y datos clínicos
# ---------------------------------------------------------------------------

def info_registro(base, registro):
    """Lee el encabezado y las anotaciones de latido de un registro.

    Salida: dict con las derivaciones, edad, sexo, diagnóstico, medicación,
    cantidad de latidos anotados y fracción de latidos no normales.
    """
    cfg = leer_config(base)
    ruta = DATOS / cfg['carpeta'] / registro
    encabezado = wfdb.rdheader(str(ruta))
    anot = wfdb.rdann(str(ruta), cfg['extension_anotaciones'])

    # Los comentarios del .hea tienen la forma:
    #   'Age: 62  Sex: M', 'Mixed angina', ..., 'Medications: ...',
    #   'Recorder type: ...'
    edad, sexo, medicacion = None, None, None
    diagnostico = []
    for linea in encabezado.comments:
        m = re.match(r'Age:\s*(\d+)\s+Sex:\s*(\w)', linea)
        if m:
            edad, sexo = int(m[1]), m[2]
        elif linea.startswith('Medications:'):
            medicacion = linea.split(':', 1)[1].strip()
        elif not linea.startswith('Recorder type:'):
            diagnostico.append(linea.strip())

    latidos = [s for s in anot.symbol if s in SIMBOLOS_LATIDO]
    n_latidos = len(latidos)
    frac_no_normales = sum(s != 'N' for s in latidos) / n_latidos

    return {
        'registro': registro,
        'deriv_0': encabezado.sig_name[0],
        'deriv_1': encabezado.sig_name[1],
        'edad': edad,
        'sexo': sexo,
        'diagnostico': '; '.join(diagnostico),
        'medicacion': medicacion,
        'n_latidos': n_latidos,
        'frac_latidos_no_normales': frac_no_normales,
    }


# ---------------------------------------------------------------------------
# 2. Estadística de episodios
# ---------------------------------------------------------------------------

def separacion_entre_episodios(tabla):
    """Para cada episodio, tiempo (s) desde el fin del episodio anterior en la
    misma derivación y del mismo tipo. NaN para el primero de cada grupo.
    Sirve para saber cuánta señal 'basal' hay antes de cada inicio."""
    tabla = tabla.sort_values('inicio_s')
    fin_anterior = tabla.groupby(['registro', 'derivacion', 'tipo'])['fin_s'].shift()
    return (tabla['inicio_s'] - fin_anterior).reindex(tabla.index)


def tiene_par_en_otra_derivacion(tabla):
    """Para cada episodio, True si se superpone en el tiempo con un episodio
    del mismo tipo en la otra derivación del mismo registro."""
    resultado = pd.Series(False, index=tabla.index)
    for _, grupo in tabla.groupby(['registro', 'tipo']):
        for i, ep in grupo.iterrows():
            otros = grupo[grupo['derivacion'] != ep['derivacion']]
            se_superpone = (otros['inicio_s'] < ep['fin_s']) & (otros['fin_s'] > ep['inicio_s'])
            resultado[i] = se_superpone.any()
    return resultado


def describir(serie):
    """Mediana [RIC] y rango de una serie numérica, como texto."""
    q1, q2, q3 = serie.quantile([0.25, 0.5, 0.75])
    return (f'mediana {q2:.1f} [RIC {q1:.1f}–{q3:.1f}], '
            f'rango {serie.min():.1f}–{serie.max():.1f}  (n={serie.count()})')


def resumen_episodios(tabla, registros):
    """Imprime la estadística de episodios y devuelve la cantidad de episodios
    ST y T (sin cambios de eje) por registro."""
    print('\n=== Episodios ===')
    ejes = tabla[tabla['cambio_eje']]
    reales = tabla[~tabla['cambio_eje']].copy()
    print(f'{len(reales)} episodios (+ {len(ejes)} cambios de eje, que se '
          f'excluyen de la estadística)')
    print(reales.groupby(['tipo', 'signo']).size().rename('n').to_string())

    print('\nEpisodios por nombre de derivación:')
    print(reales.pivot_table(index='nombre_derivacion', columns='tipo',
                             values='registro', aggfunc='count', fill_value=0)
          .to_string())

    reales['separacion_s'] = separacion_entre_episodios(reales)
    reales['par_otra_deriv'] = tiene_par_en_otra_derivacion(reales)
    for tipo, grupo in reales.groupby('tipo'):
        print(f'\n-- {tipo} --')
        print(f'duración (s):              {describir(grupo["duracion_s"])}')
        print(f'|desviación máx| (mV):     {describir(grupo["desviacion_max_mv"].abs())}')
        print(f'separación del anterior (s): {describir(grupo["separacion_s"])}')
        for limite in (120, 300, 600):
            frac = (grupo['separacion_s'] < limite).sum() / len(grupo)
            print(f'  episodios con < {limite} s desde el anterior: {frac:.0%}')
        print(f'con episodio simultáneo en la otra derivación: '
              f'{grupo["par_otra_deriv"].mean():.0%}')
        print(f'fin inferido (sin cierre anotado): {grupo["fin_inferido"].sum()}')

    por_registro = (reales.groupby(['registro', 'tipo']).size()
                    .unstack(fill_value=0)
                    .reindex(registros, fill_value=0)
                    .add_prefix('n_episodios_'))
    for col in por_registro:
        n_sin = (por_registro[col] == 0).sum()
        print(f'registros sin episodios {col[-2:].strip("_")}: {n_sin} de {len(registros)}')
    return por_registro, reales


def figura_episodios(reales, salida):
    """Histogramas de duración y de desviación máxima, ST contra T."""
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    bordes_dur = np.logspace(np.log10(25), np.log10(reales['duracion_s'].max()), 30)
    # Las desviaciones anotadas son múltiplos de 0,05 mV: los bins se centran
    # en esos valores para que ninguno caiga justo en un borde.
    bordes_desv = np.arange(-1.025, 1.05, 0.05)
    for tipo, color in (('ST', COLOR_1), ('T', COLOR_2)):
        grupo = reales[reales['tipo'] == tipo]
        estilo = dict(histtype='step', linewidth=2, color=color,
                      label=f'{tipo} (n={len(grupo)})')
        ejes[0].hist(grupo['duracion_s'], bins=bordes_dur, **estilo)
        ejes[1].hist(grupo['desviacion_max_mv'].dropna().clip(-1, 1),
                     bins=bordes_desv, **estilo)
    ejes[0].set_xscale('log')
    ejes[0].set_xlabel('Duración del episodio (s)')
    ejes[1].set_xlabel('Desviación máxima anotada (mV)  [recortada a ±1]')
    for eje in ejes:
        eje.set_ylabel('Episodios')
        eje.spines[['top', 'right']].set_visible(False)
        eje.grid(axis='y', color='#e6e6e6', linewidth=0.8)
        eje.set_axisbelow(True)
    ejes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Calidad de señal
# ---------------------------------------------------------------------------

def estado_anotado(anot, n_muestras, mascara):
    """Vector booleano (una entrada por muestra) que vale True donde el último
    '~' anterior tiene encendido el bit `mascara`. Antes del primer '~' se
    asume señal limpia."""
    estado = np.zeros(n_muestras, dtype=bool)
    marcas = [(idx, sub) for idx, sim, sub
              in zip(anot.sample, anot.symbol, anot.subtype) if sim == '~']
    # Cada tramo va desde una '~' hasta la siguiente (o el final).
    for k, (idx, sub) in enumerate(marcas):
        fin = marcas[k + 1][0] if k + 1 < len(marcas) else n_muestras
        estado[idx:fin] = bool(sub & mascara)
    return estado


def mesetas_en_extremo(x_dig, tolerancia, min_muestras):
    """Marca las muestras que forman mesetas pegadas al extremo del canal.

    Entrada: x_dig, señal en unidades del ADC (enteros). Una muestra está 'en
    el extremo' si dista <= `tolerancia` cuentas del máximo o del mínimo del
    canal. Un pico verdadero toca el extremo 1 o 2 muestras; un recorte del
    amplificador produce mesetas, así que solo se cuentan rachas de al menos
    `min_muestras` muestras seguidas.
    Salida: vector booleano del largo de x_dig.
    """
    en_extremo = (x_dig <= x_dig.min() + tolerancia) | (x_dig >= x_dig.max() - tolerancia)
    # Inicio y fin de cada racha de True.
    bordes = np.diff(np.r_[0, en_extremo.astype(int), 0])
    inicios = np.flatnonzero(bordes == 1)
    fines = np.flatnonzero(bordes == -1)
    meseta = np.zeros_like(en_extremo)
    for i, f in zip(inicios, fines):
        if f - i >= min_muestras:
            meseta[i:f] = True
    return meseta


def en_ventanas(x, n_ventana):
    """Reacomoda un vector en una matriz (ventanas x muestras), descartando
    las muestras finales que no completan una ventana."""
    n = len(x) // n_ventana
    return x[:n * n_ventana].reshape(n, n_ventana)


def calidad_registro(base, registro, p):
    """Calcula las métricas de calidad por ventana y derivación.

    Entradas: base, registro y p (parámetros de config/calidad.json).
    Salida: DataFrame con una fila por (derivación, ventana): inicio_s,
    deriva_pp_mv, ruido_af_mv, muestras_saturadas, frac_ruidosa_anot,
    frac_ilegible_anot.
    """
    cfg = leer_config(base)
    ruta = DATOS / cfg['carpeta'] / registro
    rec = wfdb.rdrecord(str(ruta), physical=False)
    anot = wfdb.rdann(str(ruta), cfg['extension_anotaciones'])
    fs = rec.fs
    n_ventana = int(round(p['ventana_s'] * fs))

    lb = p['linea_base']
    n_med_1 = muestras_impar(lb['mediana_1_s'], fs)
    n_med_2 = muestras_impar(lb['mediana_2_s'], fs)
    sos_af = butter(p['ruido_af']['orden'], p['ruido_af']['corte_hz'],
                    btype='highpass', fs=fs, output='sos')

    filas = []
    for k, nombre in enumerate(rec.sig_name):
        x_dig = rec.d_signal[:, k]
        # Conversión a mV: (cuentas - cero del ADC) / ganancia (cuentas/mV).
        x_mv = (x_dig - rec.baseline[k]) / rec.adc_gain[k]

        # Línea de base: la mediana de 200 ms saca el QRS y la de 600 ms saca
        # las ondas P y T; lo que queda es la oscilación lenta.
        linea_base = median_filter(median_filter(x_mv, n_med_1), n_med_2)
        # Ruido de alta frecuencia: filtrado de fase cero para no correr la
        # señal; el desvío robusto (MAD) no se infla con los QRS, que ocupan
        # una fracción chica de la ventana.
        alta = en_ventanas(sosfiltfilt(sos_af, x_mv), n_ventana)
        mad = np.median(np.abs(alta - np.median(alta, axis=1, keepdims=True)), axis=1)

        lb_v = en_ventanas(linea_base, n_ventana)
        meseta = mesetas_en_extremo(x_dig, p['saturacion']['tolerancia_cuentas'],
                                    p['saturacion']['meseta_min_muestras'])
        ruidosa = estado_anotado(anot, len(x_dig), BIT_RUIDOSA[k])
        ilegible = estado_anotado(anot, len(x_dig), BIT_ILEGIBLE[k])

        n = lb_v.shape[0]
        filas.append(pd.DataFrame({
            'registro': registro,
            'derivacion': k,
            'nombre_derivacion': nombre,
            'ventana': np.arange(n),
            'inicio_s': np.arange(n) * n_ventana / fs,
            'deriva_pp_mv': lb_v.max(axis=1) - lb_v.min(axis=1),
            'ruido_af_mv': 1.4826 * mad,
            'muestras_saturadas': en_ventanas(meseta, n_ventana).sum(axis=1),
            'frac_ruidosa_anot': en_ventanas(ruidosa, n_ventana).mean(axis=1),
            'frac_ilegible_anot': en_ventanas(ilegible, n_ventana).mean(axis=1),
        }))
    return pd.concat(filas, ignore_index=True)


def resumen_calidad(ventanas):
    """Imprime la distribución de las métricas y las compara entre ventanas
    anotadas como limpias y como ruidosas por la base.

    El AUC (probabilidad de que una ventana ruidosa al azar tenga una métrica
    mayor que una limpia al azar) mide si la métrica separa los dos grupos:
    0,5 = no separa, 1 = separa perfecto. Se calcula con la U de Mann-Whitney.
    """
    print('\n=== Calidad de señal ===')
    limpia = (ventanas['frac_ruidosa_anot'] == 0) & (ventanas['frac_ilegible_anot'] == 0)
    ruidosa = ventanas['frac_ruidosa_anot'] >= 0.5
    ilegible = ventanas['frac_ilegible_anot'] >= 0.5
    print(f'{len(ventanas)} ventanas-derivación: {limpia.mean():.1%} limpias según '
          f'la anotación, {ruidosa.mean():.1%} ruidosas (>= 50 % de la ventana), '
          f'{ilegible.mean():.1%} ilegibles')
    print(f'ventanas con saturación: {(ventanas["muestras_saturadas"] > 0).sum()}')

    percentiles = [0.5, 0.9, 0.95, 0.99]
    for metrica in ('deriva_pp_mv', 'ruido_af_mv'):
        print(f'\n{metrica}')
        for nombre, grupo in (('limpias', limpia), ('ruidosas', ruidosa),
                              ('ilegibles', ilegible)):
            q = ventanas.loc[grupo, metrica].quantile(percentiles)
            print(f'  {nombre:9s} ' + '  '.join(f'p{int(pc * 100)}={v:.3f}'
                                                 for pc, v in q.items()))
        u = mannwhitneyu(ventanas.loc[ruidosa, metrica],
                         ventanas.loc[limpia, metrica]).statistic
        print(f'  AUC ruidosa vs limpia: {u / (ruidosa.sum() * limpia.sum()):.2f}')


def figura_calidad(ventanas, salida):
    """Distribución de cada métrica en ventanas anotadas limpias y ruidosas."""
    limpia = (ventanas['frac_ruidosa_anot'] == 0) & (ventanas['frac_ilegible_anot'] == 0)
    ruidosa = (ventanas['frac_ruidosa_anot'] >= 0.5) | (ventanas['frac_ilegible_anot'] >= 0.5)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for eje, metrica, titulo in (
            (ejes[0], 'deriva_pp_mv', 'Deriva de línea de base, pico a pico (mV)'),
            (ejes[1], 'ruido_af_mv', 'Ruido > 40 Hz, desvío robusto (mV)')):
        valores = ventanas[metrica].clip(lower=1e-3)
        bordes = np.logspace(np.log10(valores.quantile(0.001)),
                             np.log10(valores.max()), 50)
        for grupo, color, etiqueta in ((limpia, COLOR_1, 'anotada limpia'),
                                       (ruidosa, COLOR_2, 'anotada ruidosa o ilegible')):
            # Pesos 1/n: cada curva suma 1, así se comparan grupos de distinto
            # tamaño (con eje x logarítmico, density=True deformaría la forma).
            pesos = np.full(grupo.sum(), 1 / grupo.sum())
            eje.hist(valores[grupo], bins=bordes, weights=pesos, histtype='step',
                     linewidth=2, color=color, label=f'{etiqueta} (n={grupo.sum()})')
        eje.set_xscale('log')
        eje.set_xlabel(titulo)
        eje.set_ylabel('Fracción de ventanas')
        eje.spines[['top', 'right']].set_visible(False)
        eje.grid(axis='y', color='#e6e6e6', linewidth=0.8)
        eje.set_axisbelow(True)
    # Leyenda común debajo de los dos paneles, para no tapar las curvas.
    fig.legend(*ejes[0].get_legend_handles_labels(), frameon=False,
               loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def marcar_descarte(ventanas, p):
    """Agrega la columna booleana 'descartar' a la tabla de ventanas.

    Criterio elegido (opción 1): se descarta la ventana si la fracción
    anotada como ruidosa o ilegible por la base supera
    p['descarte']['frac_anotada_max'] (0 = cualquier tramo anotado).
    Las métricas propias (deriva, ruido, saturación) se guardan pero no se
    usan para descartar: comparadas con la anotación, descartaban
    derivaciones legibles con ruido moderado constante (p. ej. e0147).
    """
    limite = p['descarte']['frac_anotada_max']
    ventanas['descartar'] = ((ventanas['frac_ruidosa_anot'] > limite)
                             | (ventanas['frac_ilegible_anot'] > limite))
    return ventanas


def resumen_por_registro(ventanas):
    """Una fila por registro con métricas de calidad de cada derivación:
    mediana y p95 de deriva y ruido, fracción del tiempo anotada como
    ruidosa o ilegible y muestras saturadas."""
    agregado = ventanas.groupby(['registro', 'derivacion']).agg(
        deriva_med_mv=('deriva_pp_mv', 'median'),
        deriva_p95_mv=('deriva_pp_mv', lambda s: s.quantile(0.95)),
        ruido_med_mv=('ruido_af_mv', 'median'),
        ruido_p95_mv=('ruido_af_mv', lambda s: s.quantile(0.95)),
        frac_ruidosa_anot=('frac_ruidosa_anot', 'mean'),
        frac_ilegible_anot=('frac_ilegible_anot', 'mean'),
        muestras_saturadas=('muestras_saturadas', 'sum'),
        frac_descartada=('descartar', 'mean'),
    )
    # Pasa de (registro, derivación) en filas a columnas con sufijo _d0 / _d1.
    ancho = agregado.unstack('derivacion')
    ancho.columns = [f'{metrica}_d{d}' for metrica, d in ancho.columns]
    return ancho


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    cfg = leer_config(base)
    p = leer_parametros()
    registros = (DATOS / cfg['carpeta'] / cfg['archivo_registros']).read_text().split()
    carpeta_fig = RESULTS / 'figuras'
    carpeta_fig.mkdir(parents=True, exist_ok=True)

    # 1. Derivaciones.
    info = pd.DataFrame([info_registro(base, r) for r in registros]).set_index('registro')
    print('=== Derivaciones ===')
    print('Combinaciones (deriv_0, deriv_1):')
    print(info.groupby(['deriv_0', 'deriv_1']).size()
          .sort_values(ascending=False).to_string())
    presencia = pd.concat([info['deriv_0'], info['deriv_1']]).value_counts()
    print('\nRegistros en que aparece cada derivación:')
    print(presencia.to_string())
    print(f'\nEdad: {describir(info["edad"])};  sexo: '
          f'{info["sexo"].value_counts().to_dict()}')

    # 2. Episodios.
    tabla = pd.read_csv(RESULTS / f'episodios_{base}.csv')
    por_registro, reales = resumen_episodios(tabla, registros)
    figura_episodios(reales, carpeta_fig / f'exploracion_episodios_{base}.png')

    # 3. Calidad.
    print('\nCalculando calidad de señal...')
    ventanas = pd.concat([calidad_registro(base, r, p) for r in registros],
                         ignore_index=True)
    resumen_calidad(ventanas)
    ventanas = marcar_descarte(ventanas, p)
    frac_por_deriv = ventanas.groupby(['registro', 'derivacion'])['descartar'].mean()
    print(f'\nVentanas descartadas: {ventanas["descartar"].mean():.1%}; '
          f'derivaciones con > 50 % descartado: '
          f'{frac_por_deriv[frac_por_deriv > 0.5].round(2).to_dict()}')
    figura_calidad(ventanas, carpeta_fig / f'exploracion_calidad_{base}.png')

    # Guardado.
    salida_reg = RESULTS / f'registros_{base}.csv'
    info.join(por_registro).join(resumen_por_registro(ventanas)).to_csv(
        salida_reg, float_format='%.4g')
    salida_vent = RESULTS / f'calidad_ventanas_{base}.csv.gz'
    ventanas.to_csv(salida_vent, index=False, float_format='%.4g')
    print(f'\nGuardado en {salida_reg}, {salida_vent} y {carpeta_fig}')
