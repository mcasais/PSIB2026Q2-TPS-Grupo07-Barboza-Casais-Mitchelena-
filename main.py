"""Ejecuta el pipeline completo, en orden, para reproducir todos los resultados.

Uso (desde cualquier carpeta):
    python main.py [base]                 # pipeline completo (base por defecto: edb)
    python main.py [base] --desde qrs     # retoma desde un paso (usa lo ya generado antes)
    python main.py --listar               # muestra los pasos, sus insumos y sus salidas

Cada paso lee lo que generó el anterior. Antes de correr un paso se comprueba que
existan sus insumos y, si falta alguno, se detiene con un mensaje claro en lugar
de fallar adentro del script. Si un paso falla, el programa termina con código 1.

Orden y dependencias (el filtrado NO va primero: necesita las detecciones QRS):
    descargar_datos -> data/<base>/
    episodios       -> episodios            (lee las anotaciones de data/)
    exploracion     -> registros, calidad   (necesita episodios)
    qrs             -> qrs, validacion_qrs  (necesita calidad)
    filtrado        -> filtrado_*           (necesita qrs, calidad, episodios)
    st_serie        -> st_serie             (necesita qrs, calidad y las funciones de filtrado)
    segmentacion    -> segmentacion_*       (necesita st_serie, episodios)
    ensamble        -> ensamble_*           (necesita st_serie, episodios)
    hrv             -> hrv                  (necesita qrs, calidad, episodios)
    estadistica     -> estadistica_*        (necesita hrv y episodios)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# (nombre, script, ¿recibe la base como argumento?, insumos, salidas)
# Los insumos y salidas son rutas relativas a la raíz del repo; {base} se
# reemplaza por la base elegida. Los datos crudos quedan en data/<base>/.
PIPELINE = [
    ("descargar_datos", "src/descargar_datos.py", False,
     [], ["data/{base}/RECORDS"]),
    ("episodios", "src/episodios.py", True,
     ["data/{base}/RECORDS"], ["results/episodios_{base}.csv"]),
    ("exploracion", "src/exploracion.py", True,
     ["data/{base}/RECORDS", "results/episodios_{base}.csv"],
     ["results/registros_{base}.csv", "results/calidad_ventanas_{base}.csv.gz"]),
    ("qrs", "src/qrs.py", True,
     ["data/{base}/RECORDS", "results/calidad_ventanas_{base}.csv.gz"],
     ["results/qrs_{base}.csv.gz", "results/validacion_qrs_{base}.csv"]),
    ("filtrado", "src/filtrado.py", True,
     ["results/qrs_{base}.csv.gz", "results/calidad_ventanas_{base}.csv.gz",
      "results/episodios_{base}.csv"],
     ["results/filtrado_real_episodios_{base}.csv"]),
    ("st_serie", "src/st_serie.py", True,
     ["results/qrs_{base}.csv.gz", "results/calidad_ventanas_{base}.csv.gz",
      "results/episodios_{base}.csv"],
     ["results/st_serie_{base}.csv.gz"]),
    ("segmentacion", "src/segmentacion.py", True,
     ["results/st_serie_{base}.csv.gz", "results/episodios_{base}.csv"],
     ["results/segmentacion_{base}.csv"]),
    ("ensamble", "src/ensamble.py", True,
     ["results/st_serie_{base}.csv.gz", "results/episodios_{base}.csv"],
     ["results/ensamble_{base}.csv", "results/ensamble_perfil_st_{base}.csv"]),
    ("hrv", "src/hrv.py", True,
     ["results/qrs_{base}.csv.gz", "results/calidad_ventanas_{base}.csv.gz",
      "results/episodios_{base}.csv"],
     ["results/hrv_{base}.csv"]),
    ("estadistica", "src/estadistica.py", True,
     ["results/hrv_{base}.csv", "results/episodios_{base}.csv"],
     ["results/estadistica_friedman_{base}.csv", "results/estadistica_episodios_comparacion_{base}.csv"]),
]
NOMBRES = [paso[0] for paso in PIPELINE]


def faltantes(rutas, base):
    """Insumos (rutas relativas a la raíz) que todavía no existen."""
    return [r.format(base=base) for r in rutas if not (ROOT / r.format(base=base)).exists()]


def listar(base):
    for i, (nombre, script, _, entradas, salidas) in enumerate(PIPELINE, 1):
        print(f"{i:2d}. {nombre:20s} {script}")
        print(f"      lee:   {', '.join(e.format(base=base) for e in entradas) or '-'}")
        print(f"      crea:  {', '.join(s.format(base=base) for s in salidas)}")


def ejecutar_pipeline(base, desde):
    """Corre los pasos desde `desde` hasta el final. Devuelve 0 si todo salió
    bien y 1 si un paso falló o le faltaban insumos."""
    (RESULTS / "figuras").mkdir(parents=True, exist_ok=True)
    for nombre, script, recibe_base, entradas, salidas in PIPELINE[NOMBRES.index(desde):]:
        print(f"\n==========================================")
        print(f"▶ Ejecutando: {script}")
        print(f"==========================================", flush=True)

        faltan = faltantes(entradas, base)
        if faltan:
            print(f"❌ Faltan insumos para '{nombre}': {', '.join(faltan)}")
            print("   Corré los pasos anteriores (sin --desde) para generarlos.")
            return 1

        comando = [sys.executable, str(ROOT / script)] + ([base] if recibe_base else [])
        # cwd = raíz del repo, así el programa funciona sin importar desde dónde se lo llame
        if subprocess.run(comando, cwd=ROOT).returncode != 0:
            print(f"❌ Error en {script}. Se detiene la ejecución.")
            return 1

        faltan = faltantes(salidas, base)
        if faltan:
            print(f"❌ '{nombre}' terminó pero no generó: {', '.join(faltan)}")
            return 1

    print("\n✅ ¡Pipeline completada con éxito!")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ejecuta el pipeline completo en orden.")
    ap.add_argument("base", nargs="?", default="edb", help="base de datos (por defecto: edb)")
    ap.add_argument("--desde", choices=NOMBRES, default=NOMBRES[0],
                    help="retoma el pipeline desde este paso")
    ap.add_argument("--listar", action="store_true", help="muestra los pasos y termina")
    args = ap.parse_args()

    if args.listar:
        listar(args.base)
        sys.exit(0)
    sys.exit(ejecutar_pipeline(args.base, args.desde))
