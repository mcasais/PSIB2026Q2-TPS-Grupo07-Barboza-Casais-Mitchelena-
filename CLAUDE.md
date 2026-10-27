# TPS Grupo 07 — Detección de episodios del segmento ST

Trabajo práctico cuatrimestral de señales de 16.63 Procesamiento de Señales e Imágenes Biomédicas (ITBA, 2C 2026).
Integrantes: Tobías Barboza, Martina Casais, Dolores Mitchelena.

## Objetivo

Detectar y caracterizar episodios de desplazamiento del segmento ST en registros ambulatorios de ECG, usando filtrado, detección de QRS, segmentación por coeficiente de disimilitud y promediado por ensamble.

Preguntas del trabajo:
- **P1.** ¿La segmentación por coeficiente de disimilitud recupera el inicio y el fin de los episodios anotados? Métricas: sensibilidad, VPP y error de localización temporal.
- **P2.** ¿Qué perfil promedio (ensamble) tiene el nivel del ST alineado con el inicio de los episodios?
- **P3.** ¿Cambian la FC y su variabilidad (bandas VLF, LF y HF) antes, durante y después de los episodios?

## Datos

- **Base:** European ST-T Database (EDB), PhysioNet, `edb` versión 1.0.0.
  - 90 registros de 2 h, de 79 sujetos.
  - 2 derivaciones por registro, a 250 Hz.
- **Archivos por registro:**
  - `.hea`: encabezado.
  - `.dat`: señal.
  - `.atr`: anotaciones de latido y de episodios.
- **Marcadores de episodio:** están en `aux_note` del `.atr`. Por ejemplo, `(ST0+` … `ST0+)` indica derivación 0, elevación. Los episodios de onda T usan `T` en lugar de `ST`.
- **Descarga:** `python src/descargar_datos.py` baja los archivos desde el espejo de AWS (`physionet-open`), porque el servidor de PhysioNet es muy lento.
- **Ubicación:** los datos van en `data/edb/` y **no se suben al repo**, porque `data/` está en `.gitignore`.
- **Long-Term ST Database (LTST):** por ahora no se usa, pero podría sumarse más adelante como validación externa. Por eso el código no debe depender de la EDB: el parser de anotaciones recibe la base como parámetro, y los parámetros (fs, derivaciones, umbrales) van en `config/`.

## Estructura del repo

```
data/edb/        datos crudos (ignorado por git)
src/             código (.py)
  rutas.py       rutas absolutas del proyecto (ROOT, DATA, RESULTS)
config/          parámetros
results/         figuras y tablas generadas
```

## Convenciones

- Scripts `.py` en `src/`, ejecutados desde la raíz del repo: `python src/<script>.py`.
- Rutas siempre importadas desde `src/rutas.py`, nunca escritas a mano.
- Unidades: amplitudes en mV, tiempos en segundos. Si una variable está en muestras, indicarlo en el nombre (por ejemplo `idx_qrs`, `t_s`).
- Funciones con docstring breve: qué hace, entradas, salidas y unidades.
- Código y comentarios en español.
- Cada integrante trabaja en su rama; los cambios se integran a `main` por pull request.

## Pipeline y estado

1. [x] Descarga de la base.
2. [ ] Parser de episodios → tabla con registro, derivación, tipo (ST/T, +/−), inicio, fin, duración y desviación máxima. Es el ground truth.
3. [ ] Exploración y acondicionamiento de la base:
   - Derivaciones por registro.
   - Estadística de episodios.
   - Calidad de señal: deriva, ruido, saturación.
4. [ ] Filtrado:
   - Pasaaltos ≤ 0,05 Hz con fase cero (Kligfield 2007), para no distorsionar el ST.
   - Alternativa: línea de base por splines.
   - Filtro óptimo (Wiener).
   - Evaluar el efecto de cada filtro sobre la medición del ST.
5. [ ] Detección QRS (Pan-Tompkins). Validar contra las anotaciones con tolerancia de unos 150 ms: sensibilidad y VPP.
6. [ ] Serie ST latido a latido por derivación:
   - Punto isoeléctrico en el segmento PR.
   - Medición en J+80 ms, o J+60 ms con FC alta.
   - Promediar varios latidos.
   - Descartar tramos de mala calidad.
   - Interpolar a una grilla uniforme.
7. [ ] Segmentación por coeficiente de disimilitud (ventana de referencia y ventana de testeo, umbral η). Evaluar contra las anotaciones con un criterio de coincidencia definido de antemano. Separar registros de ajuste y de evaluación.
8. [ ] Ensamble del nivel ST alineado al inicio de los episodios.
9. [ ] VFC: serie RR sin ectópicos, periodograma de Lomb, bandas VLF, LF y HF. VLF no es estimable en ventanas cortas y LF requiere al menos unos 2 min, así que se usan ventanas fijas alrededor del inicio.
10. [ ] Estadística, pedida por la docente:
    - Datos apareados: Friedman y Wilcoxon de rangos con signo.
    - Logaritmo de las potencias.
    - Corrección de Holm.
    - Considerar la no independencia (varios episodios por sujeto).
    - Reportar el tamaño del efecto.

## Cómo trabajar conmigo (Claude)

- Explicar cada decisión de implementación. Los integrantes tienen que poder justificar todo el código ante los docentes.
- Preferir código claro y comentado antes que compacto o "ingenioso".
- No agregar archivos de `data/` al repo ni hacer push a `main`.
- Ante una decisión metodológica no especificada en este archivo (umbrales, ventanas, criterios), proponer opciones y preguntar antes de elegir.
- Actualizar la sección "Pipeline y estado" cuando se complete una etapa.
