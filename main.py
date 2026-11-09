import subprocess
import sys

# Lista de scripts en el orden exacto del flujo de procesamiento
pipeline = [
    "src/filtrado.py",
    "src/qrs.py",
    "src/segmentacion.py",
    "src/hrv.py",
    "src/ensamble.py"
]

def ejecutar_pipeline():
    for script in pipeline:
        print(f"\n==========================================")
        print(f"▶ Ejecutando: {script}")
        print(f"==========================================")
        
        # Ejecuta el script actual y espera a que termine antes de pasar al siguiente
        resultado = subprocess.run([sys.executable, script])
        
        if resultado.returncode != 0:
            print(f"❌ Error en {script}. Se detiene la ejecución.")
            break
    else:
        print("\n✅ ¡Pipeline completada con éxito!")

if __name__ == "__main__":
    ejecutar_pipeline()
    