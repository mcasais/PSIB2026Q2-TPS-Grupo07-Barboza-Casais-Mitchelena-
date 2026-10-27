"""Parser de episodios ST y T anotados (ground truth del paso 2).

Formato de las anotaciones (documentación de la EDB, annotations.shtml):
    '(ST0+'      inicio del episodio: tipo ST, derivación 0, elevación
    'AST0+200'   pico: desviación máxima en microvoltios (200 µV)
    'ST0+)'      fin del episodio
Variantes:
    - 'T' en lugar de 'ST': episodio de onda T.
    - '++' / '--' (solo T): sub-intervalo extremo (> 0,4 mV) DENTRO de un
      episodio T. No es un episodio nuevo, así que se ignora.
    - Minúsculas ('(st0-', 'at1+350', ...): cambio de eje por cambio de
      posición. Parece un episodio pero no es isquemia; se marca con
      cambio_eje=True.

Uso (desde la raíz del repo):
    python src/episodios.py [base]      # base por defecto: edb
Salida: results/episodios_<base>.csv
"""
import json
import re
import sys

import pandas as pd
import wfdb

from rutas import CONFIG, DATOS, RESULTS

# Expresión regular de un marcador de episodio. Grupos:
#   abre   '('           -> inicio
#   pico   'A' / 'a'     -> pico
#   tipo   'ST' / 'T'    (también en minúscula)
#   deriv  un dígito     -> número de señal en el .hea
#   signo  '+', '-', '++' o '--'
#   cierra ')'           -> fin
#   valor  dígitos       -> desviación del pico en µV
# Se exige el dígito de derivación para no confundir con '(T', que es una
# anotación de ritmo (trigeminismo ventricular).
PATRON_MARCADOR = re.compile(
    r'(?P<abre>\()?(?P<pico>[Aa])?(?P<tipo>ST|st|T|t)(?P<deriv>\d)'
    r'(?P<signo>\+\+|--|\+|-)(?P<cierra>\))?(?P<valor>\d+)?'
)


def leer_config(base):
    """Devuelve el diccionario de configuración de `base` (p. ej. 'edb')
    leído de config/bases.json."""
    with open(CONFIG / 'bases.json', encoding='utf-8') as f:
        return json.load(f)[base]


def limpiar_texto(aux):
    """Corta el texto de la anotación en el primer carácter nulo.
    Algunos textos del .atr traen bytes basura después de '\\x00'
    (p. ej. '(T0+\\x00\\x13')."""
    return aux.split('\x00')[0].strip()


def interpretar_marcador(texto):
    """Interpreta el texto de una anotación de episodio.

    Entrada: texto ya limpio (str).
    Salida: dict con rol ('inicio', 'pico' o 'fin'), tipo ('ST'/'T'),
    derivacion (int), signo ('+'/'-'), extremo (bool, True si es '++'/'--'),
    cambio_eje (bool, True si está en minúscula) y valor_uv (int o None).
    Devuelve None si el texto no es un marcador de episodio.
    """
    m = PATRON_MARCADOR.fullmatch(texto)
    if m is None:
        return None

    # Exactamente una de las tres formas: '(' al principio, 'A' al
    # principio con valor al final, o ')' al final.
    if m['abre'] and not (m['pico'] or m['cierra'] or m['valor']):
        rol = 'inicio'
    elif m['pico'] and m['valor'] and not (m['abre'] or m['cierra']):
        rol = 'pico'
    elif m['cierra'] and not (m['abre'] or m['pico'] or m['valor']):
        rol = 'fin'
    else:
        return None

    return {
        'rol': rol,
        'tipo': m['tipo'].upper(),
        'derivacion': int(m['deriv']),
        'signo': m['signo'][0],
        'extremo': len(m['signo']) == 2,
        'cambio_eje': m['tipo'].islower(),
        'valor_uv': int(m['valor']) if m['valor'] else None,
    }


def parsear_registro(base, registro):
    """Arma la lista de episodios de un registro.

    Entradas: base (str, clave de config/bases.json) y registro (str, p. ej.
    'e0103').
    Salida: lista de dicts, uno por episodio, con tiempos en segundos y
    desviación máxima en mV (con signo: negativa para depresión).
    Los episodios sin cierre se cierran al final del registro con
    fin_inferido=True.
    """
    cfg = leer_config(base)
    ruta = DATOS / cfg['carpeta'] / registro
    encabezado = wfdb.rdheader(str(ruta))
    anot = wfdb.rdann(str(ruta), cfg['extension_anotaciones'])
    fs = encabezado.fs                              # Hz
    duracion_registro_s = encabezado.sig_len / fs

    # Episodios abiertos todavía sin fin. La clave identifica al episodio:
    # puede haber un ST y un T abiertos a la vez, o uno en cada derivación.
    abiertos = {}
    episodios = []

    def cerrar(clave, ep, fin_s, inferido):
        episodios.append({
            'registro': registro,
            'derivacion': clave[1],
            'nombre_derivacion': encabezado.sig_name[clave[1]],
            'tipo': clave[0],
            'signo': clave[2],
            'inicio_s': ep['inicio_s'],
            'fin_s': fin_s,
            'duracion_s': fin_s - ep['inicio_s'],
            'pico_s': ep['pico_s'],
            'desviacion_max_mv': ep['desviacion_max_mv'],
            'fin_inferido': inferido,
            'cambio_eje': clave[3],
        })

    for idx_muestra, aux in zip(anot.sample, anot.aux_note):
        marcador = interpretar_marcador(limpiar_texto(aux))
        if marcador is None or marcador['extremo']:
            continue

        t_s = idx_muestra / fs
        clave = (marcador['tipo'], marcador['derivacion'],
                 marcador['signo'], marcador['cambio_eje'])

        if marcador['rol'] == 'inicio':
            if clave in abiertos:
                print(f'  aviso {registro}: {clave} se abre dos veces '
                      f'({t_s:.1f} s); se descarta la primera apertura')
            abiertos[clave] = {'inicio_s': t_s, 'pico_s': None,
                               'desviacion_max_mv': None}

        elif marcador['rol'] == 'pico':
            if clave not in abiertos:
                print(f'  aviso {registro}: pico {clave} fuera de episodio '
                      f'({t_s:.1f} s); se ignora')
                continue
            signo = 1 if marcador['signo'] == '+' else -1
            abiertos[clave]['pico_s'] = t_s
            abiertos[clave]['desviacion_max_mv'] = signo * marcador['valor_uv'] / 1000

        else:  # fin
            if clave not in abiertos:
                print(f'  aviso {registro}: fin {clave} sin inicio '
                      f'({t_s:.1f} s); se ignora')
                continue
            cerrar(clave, abiertos.pop(clave), t_s, inferido=False)

    # Episodios que siguen abiertos al terminar el registro.
    for clave, ep in abiertos.items():
        cerrar(clave, ep, duracion_registro_s, inferido=True)

    return episodios


def parsear_base(base):
    """Recorre todos los registros de `base` y devuelve un DataFrame con un
    episodio por fila, ordenado por registro e inicio."""
    cfg = leer_config(base)
    lista = DATOS / cfg['carpeta'] / cfg['archivo_registros']
    registros = lista.read_text().split()

    episodios = []
    for registro in registros:
        episodios.extend(parsear_registro(base, registro))

    tabla = pd.DataFrame(episodios)
    return tabla.sort_values(['registro', 'inicio_s']).reset_index(drop=True)


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'edb'
    tabla = parsear_base(base)

    RESULTS.mkdir(parents=True, exist_ok=True)
    salida = RESULTS / f'episodios_{base}.csv'
    tabla.to_csv(salida, index=False)

    # Resumen para controlar contra la documentación de la base.
    reales = tabla[~tabla['cambio_eje']]
    print(f'{len(tabla)} episodios en {tabla["registro"].nunique()} registros')
    print(reales.groupby(['tipo', 'signo']).size().to_string())
    print(f'cambios de eje: {tabla["cambio_eje"].sum()}  '
          f'fin inferido: {tabla["fin_inferido"].sum()}  '
          f'sin pico: {tabla["desviacion_max_mv"].isna().sum()}')
    print(f'Guardado en {salida}')
