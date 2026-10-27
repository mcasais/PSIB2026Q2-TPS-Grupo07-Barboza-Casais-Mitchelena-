import wfdb
from rutas import DATA

DATA.mkdir(parents=True, exist_ok=True)
registros = wfdb.get_record_list('edb')
extensiones = ['hea', 'dat', 'atr']

def completo(r):
    return all((DATA / f'{r}.{e}').exists() for e in extensiones)

for r in registros:
    if completo(r):
        continue
    for intento in range(3):
        try:
            wfdb.dl_database('edb', dl_dir=str(DATA), records=[r], overwrite=True)
            print(f'{r} OK')
            break
        except Exception as err:
            print(f'{r} falló (intento {intento + 1}): {err}')

faltantes = [r for r in registros if not completo(r)]
print(f'\nCompletos: {len(registros) - len(faltantes)}/{len(registros)}')
if faltantes:
    print('Faltan:', faltantes)