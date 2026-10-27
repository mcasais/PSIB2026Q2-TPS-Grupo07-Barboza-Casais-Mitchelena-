import urllib.request
from rutas import DATA

BASE = 'https://physionet-open.s3.amazonaws.com/edb/1.0.0/'
EXT = ['hea', 'dat', 'atr']
DATA.mkdir(parents=True, exist_ok=True)

def bajar(nombre, intentos=3):
    destino = DATA / nombre
    for i in range(intentos):
        try:
            with urllib.request.urlopen(BASE + nombre, timeout=30) as r:
                destino.write_bytes(r.read())
            return True
        except Exception as e:
            print(f'  {nombre} falló (intento {i + 1}): {e}', flush=True)
    return False

# Para descargar solo algunos registros, comentá las dos líneas siguientes
# y descomentá la lista de abajo.
bajar('RECORDS')
registros = (DATA / 'RECORDS').read_text().split()
# registros = ['e0103', 'e0104', 'e0105', 'e0106', 'e0107']

print(f'{len(registros)} registros', flush=True)

for k, r in enumerate(registros, 1):
    for e in EXT:
        archivo = DATA / f'{r}.{e}'
        if archivo.exists() and archivo.stat().st_size > 0:
            continue
        bajar(f'{r}.{e}')
    print(f'[{k}/{len(registros)}] {r}', flush=True)

faltan = [r for r in registros
          if not all((DATA / f'{r}.{e}').exists() for e in EXT)]
print('Completo' if not faltan else f'Faltan: {faltan}')