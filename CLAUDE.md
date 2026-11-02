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
2. [x] Parser de episodios → tabla con registro, derivación, tipo (ST/T, +/−), inicio, fin, duración y desviación máxima. Es el ground truth.
   - `python src/episodios.py` → `results/episodios_edb.csv`.
   - Episodios sin cierre: se cierran al final del registro con `fin_inferido=True`.
   - Cambios de eje (minúsculas): se incluyen con `cambio_eje=True`.
   - Sub-intervalos extremos T++/T−−: se ignoran.
3. [x] Exploración y acondicionamiento de la base:
   - `python src/exploracion.py` → `results/registros_edb.csv`, `results/calidad_ventanas_edb.csv.gz`, `results/figuras/exploracion_*.png`. Parámetros en `config/calidad.json`.
   - [x] Derivaciones por registro: 15 combinaciones, en distinto orden; V5 (51 registros) y MLIII (47) son las más frecuentes.
   - [x] Estadística de episodios: 368 ST y 401 T (más 21 cambios de eje). El 61 % de los ST tiene un episodio simultáneo en la otra derivación.
   - [x] Calidad de señal por ventanas de 10 s: deriva (doble mediana 200/600 ms), ruido > 40 Hz (desvío robusto), saturación (mesetas en el extremo) y ruido anotado (`~`, máscara de bits en `subtype`). Las métricas se validaron contra las anotaciones `~`: AUC 0,81 (deriva) y 0,87 (ruido).
   - [x] Criterio de descarte: solo la anotación de la base. Se descarta la ventana (columna `descartar`) si tiene cualquier tramo anotado como ruidoso o ilegible (`frac_anotada_max = 0`). Resultado: 9,5 % de las ventanas; e0148-0 se pierde entera.
     - Las métricas propias no se usan para descartar porque eliminaban derivaciones legibles con ruido moderado constante (e0147-0).
     - Pendiente para la LTST u otras bases sin anotación `~`: hará falta un criterio basado en las métricas.
     - En 137 de los 368 episodios ST hay alguna ventana descartada en ±60 s alrededor del inicio. Tenerlo en cuenta en los pasos 6, 8 y 9 (interpolar o enmascarar latidos sueltos en vez de excluir el episodio entero).
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
