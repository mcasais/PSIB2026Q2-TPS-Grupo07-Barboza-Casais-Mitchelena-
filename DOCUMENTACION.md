# Documentación del repositorio — TPS Grupo 07

Detección de episodios del segmento ST en la European ST-T Database (EDB).
16.63 Procesamiento de Señales e Imágenes Biomédicas, ITBA, 2C 2026.
Integrantes: Tobías Barboza, Martina Casais, Dolores Mitchelena.

> Este documento describe el estado de la rama `martina` (commit `a5c4a85`, "filtrado").
> Explica qué contiene cada carpeta y cada archivo, y por qué se tomó cada decisión de implementación.

---

## 1. Estructura general

```
.
├── CLAUDE.md            objetivo, pipeline, convenciones e instrucciones para trabajar con Claude
├── README.md            solo el título del repo
├── DOCUMENTACION.md     este archivo
├── .gitignore           excluye data/, .venv/ y __pycache__/
├── config/              parámetros de cada etapa (JSON)
├── data/                datos crudos de PhysioNet (NO se sube al repo)
├── src/                 código Python, un script por etapa del pipeline
└── results/             tablas (.csv / .csv.gz) y figuras (.png) generadas por los scripts
```

### Orden de ejecución

Todos los scripts se ejecutan desde la raíz del repo. Cada uno lee lo que generó el anterior.

| Paso | Comando | Lee | Genera |
|---|---|---|---|
| 1 | `python src/descargar_datos.py` | PhysioNet (AWS) | `data/edb/*` |
| 2 | `python src/episodios.py` | `.hea`, `.atr` | `results/episodios_edb.csv` |
| 3 | `python src/exploracion.py` | `.hea`, `.dat`, `.atr`, `episodios_edb.csv` | `registros_edb.csv`, `calidad_ventanas_edb.csv.gz`, `figuras/exploracion_*.png` |
| 5 | `python src/qrs.py` | `.dat`, `.atr`, `calidad_ventanas_edb.csv.gz` | `qrs_edb.csv.gz`, `validacion_qrs_edb.csv` |
| 4 | `python src/filtrado.py [base] [registros...]` | todo lo anterior | `filtrado_*_edb.csv`, `figuras/filtrado_*.png` |

**El paso 5 (QRS) se ejecuta antes que el 4 (filtrado)**, porque dos de los filtros (splines y Wiener) necesitan conocer la posición de cada latido.

### Decisiones de organización (transversales)

- **Un script por etapa, con salidas en archivos.** Cada etapa se puede volver a correr sin repetir las anteriores, y los resultados intermedios quedan disponibles para revisarlos.
- **Parámetros en `config/`, no en el código.** Los umbrales, las ventanas y las frecuencias de corte se cambian sin tocar el código, y quedan documentados en un solo lugar.
- **El código no depende de la EDB.** Todos los scripts reciben la base como argumento (por defecto `edb`) y leen su configuración de `config/bases.json`. La idea es poder sumar más adelante la Long-Term ST Database (LTST) como validación externa.
- **Rutas centralizadas en `src/rutas.py`.** Ninguna ruta se escribe a mano, así que el código funciona sin importar desde qué máquina o carpeta se clone el repo.
- **Unidades:** amplitudes en mV y tiempos en s. Las variables en muestras lo indican en el nombre (`idx_qrs`, `n_ventana`).
- **Los datos crudos no se suben** (`data/` está en `.gitignore`): pesan cientos de MB y se pueden volver a descargar con un script.

---

## 2. Por qué se hace cada paso

El trabajo responde tres preguntas:
- **P1.** ¿La segmentación por coeficiente de disimilitud recupera el inicio y el fin de los episodios anotados?
- **P2.** ¿Qué perfil promedio tiene el nivel del ST alineado con el inicio de los episodios?
- **P3.** ¿Cambian la FC y su variabilidad antes, durante y después de los episodios?

Los diez pasos forman una cadena: cada uno prepara lo que necesita el siguiente. Para cada paso se indica qué problema resuelve y qué pasaría si se omitiera.

### Paso 1 — Descarga de la base
**Para qué:** contar con los registros en forma local para poder procesarlos.
**Por qué así:** se usa un script en lugar de una descarga manual para que cualquier integrante (o un docente) pueda reproducir exactamente el mismo conjunto de datos.

### Paso 2 — Parser de episodios (ground truth)
**Para qué:** obtener la referencia contra la cual se evalúa el método, es decir, cuándo empieza y termina cada episodio según los cardiólogos que anotaron la base.
**Lo usan:**
- P1, para calcular sensibilidad, VPP y error de localización: sin una referencia no se puede saber si un episodio detectado es correcto.
- P2 y P3, porque el ensamble y las ventanas de VFC se alinean con el inicio anotado.

**Si se omitiera:** no habría forma de evaluar ni de alinear nada.

### Paso 3 — Exploración y acondicionamiento de la base
**Para qué:** conocer los datos antes de diseñar el método y detectar los tramos que no conviene analizar.
- **Derivaciones:** cada registro tiene dos derivaciones distintas (15 combinaciones). Saberlo evita agrupar resultados por "derivación 0", que no significa lo mismo en todos los registros.
- **Estadística de episodios:**
  - La duración y la amplitud de los episodios orientan la elección de las ventanas y del umbral de la segmentación (paso 7).
  - La separación entre episodios indica cuánta señal basal hay antes de cada inicio, que es la que se usa como referencia en los pasos 7 a 9.
  - La simultaneidad entre derivaciones muestra si conviene combinarlas.
- **Calidad de señal:** el ruido y la deriva producen cambios aparentes del ST y detecciones falsas de QRS. Si no se descartan esos tramos, aparecen falsos episodios y la evaluación queda sesgada por artefactos y no por el método.

### Paso 4 — Filtrado
**Para qué:** quitar lo que no es ECG sin alterar el ST.
- La **deriva de línea de base** (respiración, movimiento) desplaza el nivel de la señal, y ese desplazamiento se confunde con una elevación o depresión del ST.
- La **interferencia de 50 Hz** y el **ruido muscular** aumentan la variabilidad de la medición latido a latido.

**El problema:** el ST es un nivel casi constante dentro del latido, es decir, contenido de muy baja frecuencia. Un pasaaltos con corte demasiado alto o con fase no nula lo deforma y genera desviaciones artificiales.
**Por qué se evalúa el efecto sobre el ST y no solo el aspecto de la señal:** una señal puede verse limpia y tener el ST distorsionado. Por eso se usan la prueba de impulso de la norma, una señal sintética con un ST conocido y la comparación con las desviaciones anotadas.

### Paso 5 — Detección de QRS (Pan-Tompkins)
**Para qué:** ubicar cada latido en el tiempo.
**Lo usan:**
- El paso 6, porque el punto isoeléctrico (PR) y el punto de medición (J+80 ms) se definen respecto de la onda R.
- El paso 9, que necesita los intervalos RR para la FC y la VFC.
- Los filtros por splines y Wiener del paso 4, que usan las posiciones de los latidos.

**Por qué se valida contra las anotaciones:** los errores de detección se propagan. Un latido falso produce una medición de ST en un lugar incorrecto y un RR falso que distorsiona el espectro de VFC.

### Paso 6 — Serie ST latido a latido
**Para qué:** transformar el ECG en una única serie temporal, el nivel del ST de cada latido. Esa es la variable cuyos cambios definen un episodio, y es la entrada de los pasos 7 y 8.
- **Punto isoeléctrico en el PR:** el ST se mide respecto de un nivel cero propio de cada latido, lo que compensa la deriva que haya quedado.
- **J+80 ms (J+60 ms con FC alta):** es el punto de medición estándar. Con FC alta el ST se acorta, y a 80 ms el punto podría caer sobre la onda T.
- **Promediar varios latidos:** reduce el ruido de la medición.
- **Descartar los tramos de mala calidad:** evita incluir mediciones sobre artefactos.
- **Interpolar a una grilla uniforme:** los latidos no están equiespaciados, y la segmentación y el ensamble necesitan una serie muestreada a intervalos regulares.

### Paso 7 — Segmentación por coeficiente de disimilitud
**Para qué:** responder P1, es decir, detectar automáticamente dónde cambia el nivel del ST (inicio y fin de los episodios).
- El coeficiente compara una ventana de referencia con una ventana de testeo. Si difieren más que un umbral η, se marca un límite de segmento.
- **Criterio de coincidencia definido de antemano:** se establece antes de mirar los resultados, para no ajustarlo a posteriori de manera que favorezca al método.
- **Separar registros de ajuste y de evaluación:** si η se eligiera con los mismos registros que se usan para medir el desempeño, el resultado sería optimista (sobreajuste).

### Paso 8 — Ensamble del ST alineado al inicio
**Para qué:** responder P2. Cada episodio tiene una duración y una amplitud distintas, y además ruido propio. Al alinear todos los episodios en su inicio y promediarlos, se atenúa la variabilidad individual y aparece el perfil típico: cuánto tarda en instalarse el cambio, cuál es su forma y cómo se recupera.

### Paso 9 — Variabilidad de la frecuencia cardíaca (VFC)
**Para qué:** responder P3, es decir, ver si la regulación autonómica (simpática y parasimpática) cambia alrededor de los episodios.
- **RR sin latidos ectópicos:** un latido ectópico produce un RR corto seguido de uno largo que no refleja la regulación autonómica y agrega potencia espuria al espectro.
- **Periodograma de Lomb:** la serie RR no está muestreada uniformemente, y Lomb estima el espectro sin necesidad de interpolarla.
- **Bandas VLF, LF y HF:** se asocian con distintos mecanismos de regulación; HF refleja principalmente la actividad vagal (respiratoria).
- **Ventanas fijas:** VLF no es estimable en ventanas cortas y LF necesita al menos unos 2 minutos. Por eso no se puede usar la duración variable de cada episodio, y se fijan ventanas alrededor del inicio.

### Paso 10 — Estadística
**Para qué:** determinar si las diferencias entre antes, durante y después son reales o se explican por el azar.
- **Datos apareados (Friedman y Wilcoxon):** se compara el mismo episodio en tres condiciones. Friedman es la prueba no paramétrica para más de dos condiciones apareadas, y Wilcoxon se usa después para las comparaciones de a pares.
- **Pruebas no paramétricas:** las potencias espectrales no tienen distribución normal.
- **Logaritmo de las potencias:** las potencias son muy asimétricas, y el logaritmo las vuelve más simétricas y comparables.
- **Corrección de Holm:** al hacer muchas comparaciones (varias bandas y pares de condiciones), aumenta la probabilidad de encontrar algún resultado significativo por azar. Holm controla ese error.
- **No independencia:** un mismo sujeto aporta varios episodios, que no son observaciones independientes. Si se trataran como independientes, el resultado parecería más significativo de lo que es.
- **Tamaño del efecto:** con muchos episodios, diferencias muy chicas pueden resultar significativas. El tamaño del efecto indica si la diferencia es relevante.

---

## 3. Carpeta `data/` (no versionada)

Contiene `data/edb/` con los archivos de la European ST-T Database (PhysioNet, versión 1.0.0):

- 90 registros de 2 h, de 79 sujetos, con 2 derivaciones a 250 Hz.
- Hay tres archivos por registro:
  - `.hea`: encabezado. Incluye fs, nombres de las derivaciones, ganancia del ADC y comentarios clínicos (edad, sexo, diagnóstico, medicación).
  - `.dat`: la señal digitalizada.
  - `.atr`: anotaciones. Incluye los latidos (`N`, `V`, ...), la calidad de señal (`~`) y los episodios ST/T, que están en el campo `aux_note`.
- `RECORDS`: la lista de nombres de los registros.

La subcarpeta por base (`data/edb/`, en el futuro `data/ltst/`) permite tener varias bases a la vez.

---

## 4. Carpeta `config/`

### `bases.json`
Indica, para cada base, dónde están los archivos y cómo se llaman:
- `carpeta`: `edb`
- `extension_anotaciones`: `atr`
- `archivo_registros`: `RECORDS`

**Por qué:** otras bases usan otras extensiones de anotación (por ejemplo, la LTST tiene `.sta`/`.stb`). Así, agregar una base implica sumar una entrada acá, sin modificar el código.

### `calidad.json` (paso 3)
| Parámetro | Valor | Significado |
|---|---|---|
| `ventana_s` | 10 | duración de cada ventana de calidad |
| `linea_base.mediana_1_s` / `mediana_2_s` | 0,2 / 0,6 | doble filtro de mediana para estimar la línea de base |
| `ruido_af.corte_hz` / `orden` | 40 / 4 | pasaaltos Butterworth para aislar el ruido muscular |
| `saturacion.tolerancia_cuentas` | 1 | distancia máxima al extremo del ADC para considerar saturada una muestra |
| `saturacion.meseta_min_muestras` | 3 | largo mínimo de una meseta para considerarla saturación |
| `descarte.frac_anotada_max` | 0,0 | se descarta una ventana con cualquier tramo anotado como ruidoso o ilegible |

### `qrs.json` (paso 5)
| Parámetro | Valor | Significado |
|---|---|---|
| `pasabanda_hz` | [8, 20] | pasabanda de Pan-Tompkins (el original usa 5–15 Hz; ver sección 5.5) |
| `orden_pasabanda` | 2 | orden del Butterworth |
| `ventana_integracion_s` | 0,150 | ventana de integración móvil (ancho de un QRS ancho) |
| `refractario_s` | 0,200 | separación mínima entre dos QRS |
| `ventana_onda_t_s` | 0,360 | dentro de este intervalo se evalúa si un pico es una onda T |
| `aprendizaje_s` | 2,0 | duración de la fase inicial que fija los umbrales |
| `factor_busqueda_atras` | 1,66 | se busca un latido perdido si pasan más de 1,66 RR promedio |
| `n_rr_promedio` | 8 | cantidad de RR que se promedian |
| `reinicio_sin_qrs_s` | 3,0 | tiempo sin detecciones tras el cual se reinician los umbrales (agregado propio) |
| `tolerancia_validacion_s` | 0,150 | tolerancia para emparejar las detecciones con las anotaciones |

### `filtrado.json` (paso 4)
- **`filtros`**: define cada filtro del banco (`pa_005`, `pa_067`, `notch_50`, `pb_40`, `splines`, `wiener`).
- **`cadenas`**: combinaciones de filtros que se aplican en orden y se comparan entre sí:
  `crudo`, `pa_005`, `pa_067`, `splines`, `wiener` (= pa_005 + Wiener), `pa_005_notch_pb40`, `splines_notch_pb40`.
- **`padding_periodos`**: 5. Cuánto se extiende la señal en los bordes antes de filtrar.
- **`isoelectrico`**: el punto isoeléctrico se busca entre R−120 y R−40 ms, en un tramo de 20 ms.
- **`medicion_st`**: el ST se mide en R+120 ms, promediando 20 ms.
- **`wiener`**: latido de R−250 a R+450 ms, tramos de 60 s, NFFT 256, al menos 20 latidos por tramo.
- **`impulso`**: prueba AHA/IEC con un pulso de 3 mV × 100 ms. Límites: 0,1 mV de desplazamiento y 0,3 mV/s de pendiente.
- **`sintetica`**: parámetros de la señal sintética. Usa 3 plantillas (e0103-0, e0103-1, e0305-1) y 5 repeticiones de 600 s, con un episodio de −0,2 mV y deriva, ruido blanco y 50 Hz.
- **`real`**: la referencia son los primeros 30 s, la ventana alrededor del pico es de ±5 s y se exigen al menos 3 latidos.
- **`ejemplo`**: tramo que se usa en la figura de ejemplo (e0147, derivación 0, desde 1830 s, 20 s).

---

## 5. Carpeta `src/`

### 5.1 `rutas.py`
Define las rutas absolutas del proyecto a partir de la ubicación del propio archivo:
`ROOT` (raíz), `DATOS` (`data/`), `DATA` (`data/edb/`, la usa el script de descarga), `CONFIG` y `RESULTS`.

**Por qué:** si las rutas se escribieran a mano, el código fallaría al ejecutarlo desde otra carpeta o en otra computadora.

### 5.2 `descargar_datos.py` (paso 1)
Descarga `RECORDS` y, para cada registro, los archivos `.hea`, `.dat` y `.atr`.

Decisiones:
- **Usa el espejo de AWS (`physionet-open.s3.amazonaws.com`)** en lugar del servidor de PhysioNet, porque este último es muy lento.
- **Hace 3 intentos por archivo**, para tolerar cortes de red.
- **Saltea los archivos que ya existen y no están vacíos.** Si la descarga se corta, se puede volver a correr el script y continúa donde quedó.
- Tiene una lista comentada para descargar solo algunos registros, útil para hacer pruebas rápidas.

Limitaciones conocidas:
- Si falla la descarga de `RECORDS`, el script se detiene con un error poco claro.
- El control final de archivos faltantes no detecta archivos vacíos.

### 5.3 `episodios.py` (paso 2: ground truth)
Lee las anotaciones `.atr` y arma una tabla con un episodio por fila.

**Formato de las anotaciones de la EDB:**
- `(ST0+`: inicio de un episodio ST en la derivación 0, de elevación.
- `AST0+200`: pico del episodio, con una desviación de 200 µV.
- `ST0+)`: fin del episodio.
- Con `T` en lugar de `ST`, el episodio es de onda T.

**Funciones:**
- `leer_config(base)`: lee `config/bases.json`. La reutilizan los demás scripts.
- `limpiar_texto(aux)`: corta el texto en el primer carácter nulo.
- `interpretar_marcador(texto)`: aplica una expresión regular y devuelve el rol (inicio/pico/fin), el tipo, la derivación, el signo y el valor.
- `parsear_registro(base, registro)`: recorre las anotaciones y va abriendo y cerrando episodios.
- `parsear_base(base)`: aplica lo anterior a todos los registros.

**Decisiones:**
| Decisión | Justificación |
|---|---|
| La expresión regular exige el dígito de derivación | `(T` sin dígito es una anotación de ritmo (trigeminismo), no un episodio de onda T. |
| Se cortan los textos en `\x00` | Algunos `aux_note` traen bytes basura después del carácter nulo (por ejemplo, `(T0+\x00\x13`), y sin limpiarlos el marcador no se reconoce. |
| Se ignoran `++` / `--` | Marcan un sub-intervalo extremo (> 0,4 mV) dentro de un episodio T ya abierto, no un episodio nuevo. |
| Las minúsculas se incluyen con `cambio_eje=True` | Son cambios de eje por un cambio de posición del paciente: se parecen a un episodio pero no son isquemia. Se conservan para poder analizarlas o excluirlas. |
| La clave del episodio es (tipo, derivación, signo, cambio_eje) | Puede haber un ST y un T abiertos a la vez, o episodios simultáneos en las dos derivaciones. |
| Los episodios sin cierre se cierran al final del registro, con `fin_inferido=True` | Así no se pierde el episodio, y queda indicado que su fin no es real. |
| Si un episodio se abre dos veces, se conserva la segunda apertura | Es un error de anotación. Se imprime un aviso para poder revisarlo. |
| La desviación se pasa a mV y lleva signo (negativo = depresión) | Respeta la convención de unidades del proyecto y permite distinguir elevación de depresión con un solo número. |

**Salida:** `results/episodios_edb.csv`, con 790 episodios. Son 368 ST y 401 T, más 21 cambios de eje.

### 5.4 `exploracion.py` (paso 3: exploración y calidad)
Tiene tres partes:

**1. Derivaciones y datos clínicos** (`info_registro`)
Extrae del `.hea` las derivaciones, la edad, el sexo, el diagnóstico y la medicación. Del `.atr` obtiene la cantidad de latidos y la fracción de latidos no normales.
- Hay 15 combinaciones de derivaciones, en distinto orden. Las más frecuentes son V5 (51 registros) y MLIII (47).
- **Implicancia:** la derivación 0 no corresponde a la misma derivación en todos los registros, así que no se pueden agrupar resultados "por derivación 0".

**2. Estadística de episodios** (`resumen_episodios`, `figura_episodios`)
Calcula la duración, la desviación máxima y la separación respecto del episodio anterior, y verifica si cada episodio coincide con otro en la otra derivación.
- Se excluyen los cambios de eje.
- La separación entre episodios indica cuánta señal basal hay antes de cada inicio, un dato necesario para los pasos 7 a 9.
- Resultado: el 61 % de los episodios ST tiene un episodio simultáneo en la otra derivación.
- En la figura, los bins de desviación se centran en múltiplos de 0,05 mV porque las desviaciones anotadas son múltiplos de ese valor.

**3. Calidad de señal por ventanas de 10 s** (`calidad_registro`)
Para cada ventana y derivación se calculan:
| Métrica | Cómo se calcula | Por qué |
|---|---|---|
| `deriva_pp_mv` | pico a pico de la línea de base estimada con un doble filtro de mediana de 200 ms y 600 ms (de Chazal 2004) | la mediana de 200 ms elimina el QRS y la de 600 ms elimina P y T; lo que queda es la oscilación lenta |
| `ruido_af_mv` | 1,4826·MAD de la señal filtrada con un pasaaltos de 40 Hz con fase cero | mide el ruido muscular; el MAD es robusto y no se infla con los QRS, que ocupan poco tiempo |
| `muestras_saturadas` | mesetas de ≥ 3 muestras a ≤ 1 cuenta del extremo del ADC | un pico real toca el extremo 1 o 2 muestras; un recorte del amplificador produce mesetas |
| `frac_ruidosa_anot` / `frac_ilegible_anot` | fracción de la ventana marcada por las anotaciones `~` (máscara de bits en `subtype`: bits 0/1 = ruidosa, bits 4/5 = ilegible) | es la anotación experta de la propia base |

- Para la saturación se trabaja con la señal digital (`physical=False`), porque los extremos del ADC se identifican mejor en cuentas enteras.
- **Validación de las métricas:** se calculó el AUC de Mann-Whitney entre ventanas anotadas como ruidosas y como limpias. Dio 0,81 para la deriva y 0,87 para el ruido, así que las métricas sí separan los dos grupos.

**Criterio de descarte** (`marcar_descarte`):
- Se descarta una ventana si tiene **cualquier** tramo anotado como ruidoso o ilegible por la base (`frac_anotada_max = 0`).
- **Por qué no se usan las métricas propias:** al compararlas con la anotación, eliminaban derivaciones legibles con ruido moderado pero constante (por ejemplo, e0147-0).
- Resultado: se descarta el 9,5 % de las ventanas, y la derivación e0148-0 se pierde entera.
- Las métricas propias se guardan igual, como valores continuos, porque harán falta para bases sin anotación `~` (como la LTST).
- En 137 de los 368 episodios ST hay alguna ventana descartada en los ±60 s alrededor del inicio. En los pasos 6, 8 y 9 conviene interpolar o enmascarar latidos sueltos en lugar de excluir esos episodios enteros.

**`mascara_descarte`:** convierte la columna `descartar` (una por ventana) en un vector booleano por muestra. La usan `qrs.py` y `filtrado.py`.

### 5.5 `qrs.py` (paso 5: detección de QRS con Pan-Tompkins)
Se adelantó porque el filtrado necesita las posiciones de los latidos.

**Etapas** (Pan y Tompkins 1985):
1. Pasabanda: conserva la energía del QRS y atenúa P, T, deriva y ruido muscular.
2. Derivada de 5 puntos: resalta las pendientes empinadas.
3. Cuadrado: hace todo positivo y realza las pendientes grandes.
4. Integración móvil de 150 ms: convierte cada QRS en un solo lóbulo.
5. Umbrales adaptativos (`umbralizar`): SPKI y NPKI se actualizan con un promedio exponencial y el umbral es `THR1 = NPKI + 0,25·(SPKI − NPKI)`. Incluye un período refractario de 200 ms, la discriminación de onda T (un pico a < 360 ms con pendiente menor a la mitad de la del QRS anterior) y la búsqueda hacia atrás (si pasan > 1,66 RR sin detección, se acepta el mayor pico que supere THR1/2).

**Diferencias con el original y su justificación:**
| Cambio | Justificación |
|---|---|
| Filtros de fase cero (`sosfiltfilt`) y ventana de integración centrada | El análisis es fuera de línea: no hay retardo que compensar y el pico queda en su lugar. |
| Butterworth diseñado para la fs del registro | Los coeficientes enteros del original están pensados para 200 Hz, y la EDB está a 250 Hz. |
| **Pasabanda de 8–20 Hz en lugar de 5–15 Hz** | En derivaciones con ondas T grandes (por ejemplo, V2 de e0305), la T se detectaba como un segundo latido. La T tiene menos energía que el QRS por encima de 8 Hz. Con el cambio, sobre toda la EDB, el VPP pasó de 0,982 a 0,995, la Se de 0,9984 a 0,9987, y las derivaciones con VPP < 0,95 bajaron de 11 a 2. |
| Reinicio de umbrales tras 3 s sin detecciones | Un artefacto grande inflaba SPKI y el detector no volvía a detectar nada (pasaba en e0611). |
| Se usa un solo juego de umbrales (el de la señal integrada) y no se reducen a la mitad con RR irregular | Es la versión que se presenta en los apuntes del curso. |
| El pico del QRS se ubica en el máximo de \|señal pasabanda\| alrededor del pico de la integrada | El pico de la integrada no coincide con la onda R. |
| El detector se aplica a cada derivación por separado | El filtrado de cada derivación usa sus propios latidos. |

**Validación** (`emparejar`, `procesar_registro`):
- Se emparejan las detecciones con las anotaciones de latido, uno a uno, con una tolerancia de 150 ms.
- Se calcula en dos alcances: `todo` (el registro completo) y `sin_descartadas` (sin las ventanas descartadas en el paso 3).
- Los totales se agregan sumando VP, FP y FN de todos los registros, como indica la norma ANSI/AAMI EC57.

| Alcance | Se | VPP | Derivaciones con VPP < 0,95 |
|---|---|---|---|
| todo | 0,9969 | 0,9901 | 7 |
| sin_descartadas | 0,9987 | 0,9952 | 2 |

`leer_senal_mv(base, registro)` también se define acá y la reutiliza `filtrado.py`.

### 5.6 `filtrado.py` (paso 4: filtrado y su efecto sobre el ST)

**Banco de filtros.** Todos son de fase cero, para no desplazar ni deformar el ST.
| Filtro | Qué hace | Por qué se incluye |
|---|---|---|
| `pa_005` | pasaaltos Butterworth de 0,05 Hz, orden 2 | es el corte que recomienda Kligfield 2007 (AHA) para no distorsionar el ST |
| `pa_067` | pasaaltos de 0,67 Hz | Kligfield lo admite solo con fase cero; sirve de comparación para ver cuánto ST se pierde con un corte más alto |
| `splines` | línea de base por splines cúbicas, con un nodo por latido en el tramo isoeléctrico del PR, que se resta a la señal | alternativa no lineal que sigue la deriva sin afectar el contenido de baja frecuencia del ST |
| `wiener` | filtro óptimo de Wiener con el latido promedio como modelo de señal: H(f) = Ss/(Ss+Sn) (Rangayyan, cap. 3) | es el filtro óptimo pedido en el curso |
| `notch_50` | ranura de 50 Hz con Q = 30 | interferencia de línea, fuerte en e0147 |
| `pb_40` | pasabajos de 40 Hz, orden 4 | ruido muscular |

**Detalles de implementación:**
- **Diseño en secciones de segundo orden (SOS):** con un corte de 0,05 Hz a 250 Hz, los coeficientes b, a quedan mal condicionados numéricamente.
- **`filtfilt` (ida y vuelta):** la fase se cancela. Como el módulo queda al cuadrado, el corte a −3 dB de cada pasada queda a −6 dB.
- **Padding de 5 períodos del corte con reflexión impar:** evita que el transitorio inicial caiga sobre la señal.
- **Punto isoeléctrico** (`nivel_isoelectrico`): dentro de R−120 a R−40 ms se elige el tramo de 20 ms más plano, es decir, el de menor suma de |diferencias|. Así se evita caer sobre la onda P o el inicio del QRS. Se calcula con sumas acumuladas para no recorrer los latidos con un bucle.
- **Splines:** se excluyen los nodos en ventanas descartadas, para no apoyar la curva sobre ruido. Antes del primer nodo y después del último se mantiene constante el valor del nodo extremo.
- **Wiener**, por tramos de 60 s:
  - Sn = espectro medio de los residuos (latido − promedio), que es lo que cambia de un latido a otro.
  - Ss = espectro del promedio − Sn/N, porque el promedio de N latidos todavía conserva una varianza de ruido Sn/N.
  - H(f) es real y está entre 0 y 1, así que el filtro es de fase cero. Se pasa a un FIR simétrico de largo impar con ventana de Hann, para que `mode='same'` no desplace la señal.
  - Se agrega un margen de los tramos vecinos para que los bordes queden bien filtrados.
  - Los tramos con menos de 20 latidos válidos quedan sin filtrar.
- **Medición del ST provisoria** (`medir_st`): nivel en R+120 ms (≈ J+80 ms) menos el nivel isoeléctrico, ambos promediados en 20 ms. Alcanza para comparar filtros entre sí. **La reemplaza la medición del paso 6.**

**Evaluación (tres niveles):**

1. **Prueba de impulso AHA/IEC** (`prueba_impulso`): un pulso de 3 mV × 100 ms en 60 s de señal nula. Fuera del pulso, la salida no debe superar 0,1 mV ni 0,3 mV/s. Se deja una muestra de guarda a cada lado del pulso, porque la derivada centrada incluye el salto.
   | Filtro | Desplazamiento máx. (mV) | Pendiente máx. (mV/s) | Cumple |
   |---|---|---|---|
   | pa_005 | 0,033 | 0,005 | Sí |
   | pa_067 | 0,433 | 0,851 | **No** |

2. **Señal sintética** (`plantilla_limpia`, `senal_sintetica`, `evaluar_sintetica`): permite conocer el ST verdadero.
   - Plantilla: el latido promedio del primer minuto limpio y sin episodios de un registro real, filtrado con pa_005, al que se le resta el nivel isoeléctrico y se le aplica una ventana de Tukey para que empiece y termine en cero.
   - RR de 0,85 ± 0,05 s, con una variación sinusoidal a 0,25 Hz que imita la arritmia respiratoria.
   - Episodio: un escalón suave entre R+40 y R+400 ms, cuya amplitud sigue un trapecio en el tiempo (0 → −0,2 mV en 60 s, meseta de 120 s y vuelta en 60 s).
   - Ruido: deriva (sinusoides de 0,01–0,5 Hz con amplitud ∝ 1/f, 0,3 mV RMS), ruido blanco de 0,02 mV y 50 Hz de 0,05 mV.
   - Métricas: sesgo y RMSE del ST latido a latido, y error del cambio de ST durante la meseta. Se descartan los primeros y los últimos 20 s.
   - Resultados (con ruido, promedio de plantillas y repeticiones):
     | Cadena | RMSE (mV) | \|error del cambio\| (mV) |
     |---|---|---|
     | crudo | 0,024 | 0,002 |
     | pa_005 | 0,022 | 0,002 |
     | pa_067 | 0,023 | **0,015** |
     | splines | 0,020 | 0,002 |
     | wiener | 0,027 | **0,034** |
     | pa_005_notch_pb40 | 0,017 | 0,002 |
     | splines_notch_pb40 | **0,016** | **0,001** |

3. **Señal real** (`evaluar_real`, `resumen_real`):
   - Desviación medida = mediana del ST en ±5 s alrededor del pico anotado, menos la mediana de los primeros 30 s válidos. Se toma esta referencia porque así define la EDB la desviación anotada (respecto del comienzo del registro).
   - Ruido de la serie ST: 1,4826·MAD de la diferencia entre latidos consecutivos, dividido por √2, medido fuera de los episodios.
   - Por cadena se reportan el error absoluto medio, el sesgo, la correlación de Spearman, el acierto del signo y la mediana del ruido.
   - **En los resultados actuales, la evaluación real se corrió solo sobre e0103, e0104 y e0147** (16 episodios ST). Con esos datos, pa_005, splines y crudo dan el menor error (≈ 0,036 mV), y wiener el mayor (0,052 mV).

**Conclusión preliminar** (todavía no está registrada en `CLAUDE.md`):
- pa_067 no cumple la norma AHA/IEC y subestima el cambio de ST.
- El Wiener con el latido promedio atenúa el cambio de ST: como modela la señal con el latido promedio del tramo de 60 s, interpreta la evolución del ST como ruido.
- Las cadenas pa_005 o splines, seguidas de notch y pb40, son las que mejor conservan el ST.
- La elección final de la cadena está pendiente.

---

## 6. Carpeta `results/`

### Tablas
| Archivo | Generado por | Contenido (una fila por…) |
|---|---|---|
| `episodios_edb.csv` | `episodios.py` | episodio: registro, derivación, nombre de la derivación, tipo (ST/T), signo, inicio, fin, duración y pico (s), desviación máxima (mV), `fin_inferido`, `cambio_eje`. **Es el ground truth.** |
| `registros_edb.csv` | `exploracion.py` | registro (90): derivaciones, edad, sexo, diagnóstico, medicación, latidos, fracción de no normales, cantidad de episodios ST/T y métricas de calidad por derivación (sufijos `_d0` / `_d1`) |
| `calidad_ventanas_edb.csv.gz` | `exploracion.py` | ventana de 10 s × derivación (~129 600 filas): deriva, ruido AF, saturación, fracción anotada como ruidosa/ilegible y columna `descartar` |
| `qrs_edb.csv.gz` | `qrs.py` | latido detectado (~1,59 millones): registro, derivación, `idx_qrs` (muestra) |
| `validacion_qrs_edb.csv` | `qrs.py` | registro × derivación × alcance: VP, FP, FN, sensibilidad y VPP |
| `filtrado_impulso_edb.csv` | `filtrado.py` | filtro pasaaltos: resultado de la prueba de impulso |
| `filtrado_sintetica_edb.csv` | `filtrado.py` | plantilla × repetición × condición × cadena: sesgo, RMSE y error del cambio |
| `filtrado_real_episodios_edb.csv` | `filtrado.py` | episodio × cadena: desviación anotada y medida |
| `filtrado_real_ruido_edb.csv` | `filtrado.py` | registro × derivación × cadena: ruido latido a latido del ST |

Las tablas grandes se guardan comprimidas (`.csv.gz`). pandas las lee directamente con `pd.read_csv`.

### Figuras (`results/figuras/`)
| Figura | Muestra |
|---|---|
| `exploracion_episodios_edb.png` | histogramas de duración (escala log) y de desviación máxima, ST contra T |
| `exploracion_calidad_edb.png` | distribución de la deriva y del ruido AF en ventanas anotadas como limpias y como ruidosas |
| `filtrado_respuestas_edb.png` | respuesta en frecuencia de los pasaaltos, del pasabajos y de la ranura, y salida ante el impulso AHA/IEC |
| `filtrado_sintetica_edb.png` | RMSE y error del cambio de ST por cadena, sin y con ruido |
| `filtrado_real_edb.png` | desviación medida contra anotada (pa_005 y pa_067), error medio y ruido por cadena |
| `filtrado_ejemplo_edb.png` | tramo de 20 s de e0147 con deriva fuerte, crudo y después de cada cadena |

Criterios comunes de las figuras: paleta azul (`#2a78d6`) y naranja (`#eb6834`), ejes sin marco superior ni derecho, grilla suave y leyenda fuera de las curvas.

---

## 7. Estado del pipeline y pendientes

| Paso | Estado |
|---|---|
| 1. Descarga | Hecho |
| 2. Parser de episodios | Hecho |
| 3. Exploración y calidad | Hecho |
| 4. Filtrado | El código está completo y la comparación de filtros está hecha. Faltan dos cosas: (a) correr la comparación sobre señal real con los 90 registros, porque los resultados actuales salen de solo 3 (e0103, e0104 y e0147); (b) decidir qué cadena de filtros se usa en los pasos siguientes |
| 5. Detección QRS | Implementado y validado |
| 6. Serie ST latido a latido | Pendiente |
| 7. Segmentación por coeficiente de disimilitud | Pendiente |
| 8. Ensamble del ST alineado al inicio | Pendiente |
| 9. VFC (Lomb, VLF/LF/HF) | Pendiente |
| 10. Estadística (Friedman, Wilcoxon, Holm, tamaño del efecto) | Pendiente |

Para tener en cuenta:
- La sección "Pipeline y estado" de `CLAUDE.md` todavía marca los pasos 4 y 5 como pendientes.
- `README.md` está vacío y podría reemplazarse por una versión corta de este documento.
