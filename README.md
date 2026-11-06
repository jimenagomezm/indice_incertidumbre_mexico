# Índice de incertidumbre de Banxico, medido con texto

Un índice de la incertidumbre que expresa el Banco de México en sus **anuncios de política monetaria** (2000–2026) y en sus **minutas** (2011–2026). Cada documento se divide en chunks, un LLM califica la incertidumbre de cada pedazo, y el índice del documento es el promedio de esas calificaciones. Se compara este método con un indice construido con diccionarios y otro con embeddings.

El repo incluye los datos procesados, no hace falta descargar nada ni pagar nada para ver los resultados.

---

## 1. Inicio

```bash
# 1. Instalar uv 
curl -LsSf https://astral.sh/uv/install.sh | sh        # Mac / Linux
# Windows (PowerShell): powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Dentro de esta carpeta, instalar Python 3.13 y todos los paquetes
cd indice_banxico
uv sync

# 3. Abrir los notebooks
uv run jupyter lab
```

Después abre `notebooks/06_resultados.ipynb` y córrelo completo. Ahí están todos los resultados.

---

## 2. Instalar uv (también se puede con pip)

[uv](https://docs.astral.sh/uv/) es un administrador de Python y de paquetes. Correr `uv sync`:

- descarga Python 3.13 si no lo tienes
- crea un entorno aislado en `.venv/`, dentro de esta carpeta
- instala las mismas versiones con las que se hicieron los resultados. Están fijadas en `uv.lock`.


### Paquetes principales

| Paquete | Para qué se usa |
|---|---|
| `requests`, `lxml` | descargar las páginas y PDFs de Banxico |
| `pymupdf` | extraer el texto de los PDFs |
| `pandas`, `numpy`, `pyarrow` | tablas y archivos `.parquet` |
| `sentence-transformers`, `transformers` (instala `torch`) | Método 2, embeddings |
| `httpx`, `pydantic`, `python-dotenv` | Método 3: llamar al LLM y validar su respuesta |
| `matplotlib` | gráficas |
| `jupyter`, `ipykernel` | notebooks |

---

## 3. Estructura 

```
indice_banxico/
├── README.md                  este archivo
├── pyproject.toml             lista de paquetes (la lee uv)
├── uv.lock                    versiones exactas (las lee uv)
├── requirements.txt           mismas versiones, para pip
├── .python-version            Python 3.13
├── .env.example               plantilla para las llaves (sólo para el paso pagado)
│
├── notebooks/                 el pipeline, en orden correr de 01 a 07
├── src/                       las funciones que usan los notebooks
├── scripts/
│   └── score_chunks_openrouter.py   manda los pedazos al LLM (paso pagado, opcional)
├── data/processed/banxico/    todos los datos intermedios, ya calculados
└──resultados/                los dos CSV finales (y las figuras, al correr 06)
```

### `notebooks/`

| Notebook | Qué hace | Lee | Escribe |
|---|---|---|---|
| `01_corpus_anuncios` | Descarga los PDFs de los anuncios, extrae el texto, lo divide en pedazos | sitio de Banxico | `banxico_anuncios_{metadata,pages,documents,chunks}` |
| `02_corpus_minutas` | Lo mismo para las minutas | sitio de Banxico | `banxico_minutas_{...}` |
| `03_metodo1_diccionario` | Método 1: conteo de palabras | chunks | `*_index_exact_matching.parquet` |
| `04_metodo2_embeddings` | Método 2: embeddings | chunks | `*_chunk_embeddings.npy`, `*_index_embeddings.parquet` |
| `05_metodo3_llm` | Método 3: junta las calificaciones del LLM en un índice por documento | `llm_scores/` | `*_index_llm.parquet` |
| `06_resultados` | **Todos los resultados**: índice, promedio móvil, z-score, trimestral, validación contra el peso, episodios, los tres métodos. Genera los CSV y todas las figuras de la presentación | los `*_index_*.parquet` | `resultados/*.csv`, `resultados/figuras/*.png` |
| `07_demo_clase` | demo para clase: un documento → pedazos → prompt → una llamada al LLM → índice | todo lo anterior | nada |

### `src/`

| Archivo | Qué contiene |
|---|---|
| `paths.py` | las rutas de las carpetas; hace que los notebooks encuentren `src/` |
| `corpus/scraping.py` | descarga de páginas y PDFs |
| `corpus/banxico_anuncios.py` | cómo leer la página de anuncios de Banxico y armar los metadatos |
| `corpus/banxico_minutas.py` | lo mismo para las minutas |
| `corpus/processing.py` | extrae y limpia el texto de los PDFs (por página y por documento) |
| `corpus/chunking.py` | divide cada documento en chunks de aprox 3,000 caracteres |
| `index/exact_matching.py` | Método 1 (diccionario) |
| `index/embeddings.py` | Método 2 (embeddings) |
| `index/llm_prompts.py` | los prompts del LLM (calibrado y laxo, para anuncios y minutas) |
| `index/llm_scoring.py` | el formato de respuesta que se le exige al LLM |
| `index/common.py` | funciones compartidas por los métodos |
| `analysis/sie.py` | descarga el tipo de cambio del SIE de Banxico (para validar) |

### `data/processed/banxico/`

Un juego de archivos por corpus (`banxico_anuncios_*` y `banxico_minutas_*`):

| Archivo | Contenido |
|---|---|
| `*_metadata.csv` | un renglón por documento: id, fecha, título, URL del PDF |
| `*_pages.parquet` | el texto de cada página |
| `*_documents.parquet` | el texto completo de cada documento |
| `*_chunks.parquet` | los chunks |
| `*_chunk_embeddings.npy`, `*_embedding_parent_scores.parquet` | caché del Método 2 (evita recalcular) |
| `*_index_exact_matching.parquet` | índice del Método 1, uno por documento |
| `*_index_embeddings.parquet` | índice del Método 2 |
| `*_index_llm.parquet` | índice del Método 3, las 4 versiones |
| `llm_scores/*.parquet` | la calificación del LLM a cada chunks: 8 archivos = 2 corpus × 2 prompts × 2 modelos |
| `sie_exchange_rate.parquet` | tipo de cambio diario peso-dólar (SIE), para la validación |

Los IDs de documento tienen la fecha de la decisión: `banxico_apm_2020_04_21` (anuncio) y `banxico_min_2020_04_21` (minuta). Así, el anuncio y la minuta de la misma reunión coinciden.

---


## 4. Orden para correr

Todo ya está calculado, así que puedes empezar directamente en el paso 6. Los pasos 1–5 sirven para reproducir desde cero.

| Paso | Notebook | Obligatorio? | Tiempo | Notas |
|---|---|---|---|---|
| 1 | `01_corpus_anuncios` | No | ~5 min | Descarga ~20 MB de PDFs |
| 2 | `02_corpus_minutas` | No | ~10 min | Descarga ~90 MB de PDFs |
| 3 | `03_metodo1_diccionario` | No | segundos | |
| 4 | `04_metodo2_embeddings` | No | segundos (con caché) | Sin caché: descarga el modelo (~1 GB) y tarda unos minutos |
| — | `scripts/score_chunks_openrouter.py` | No, cuesta dinero | | Ver sección 5 |
| 5 | `05_metodo3_llm` | No | segundos | Lee las calificaciones guardadas, gratis |
| 6 | `06_resultados` | **Sí** | ~1 min | Escribe los CSV y las figuras |
| 7 | `07_demo_clase` | Opcional | segundos | La llamada en vivo en la demo necesita llave, sin ella usa la calificación guardada |

**Fecha de corte.** Los notebooks 01 y 02 tienen `FECHA_CORTE = "2026-09-30"`. Banxico sigue publicando documentos; con la fecha de corte, volver a descargar da exactamente los mismos documentos que se usaron en la presentación. Para agregar documentos nuevos se cambia la fecha, pero esos documentos nuevos no tendrán calificación del LLM hasta correr el paso pagado.

---

## 5. Calificar con el LLM (no necesario volverlo a hacer, todo cuesta 15 dolares aprox)

Las calificaciones del LLM ya están en `data/processed/banxico/llm_scores/`. Sólo hace falta este paso para calificar documentos nuevos o repetir el experimento.

Es necesario crear cuenta en [OpenRouter](https://openrouter.ai) y una llave y copiarla a `.env` 

Primero, sin costo, ver qué se mandaría:
```bash
uv run python scripts/score_chunks_openrouter.py \
    --corpus banxico_anuncios --variant calibrated \
    --model openai/gpt-6.1-sol --dry-run
```

Luego quitar `--dry-run` para correrlo. El script pide escribir `SPEND` para confirmar y se detiene si el costo pasa de `--max-cost`.

El script **sólo califica pedazos que no tienen calificación**, así que volver a correrlo no cobra dos veces. Opciones: `--corpus` (`banxico_anuncios`, `banxico_minutas`), `--variant` (`calibrated`, `lax`), `--model` (`openai/gpt-6.1-sol`, `openai/gpt-6-luna`).

Lo que costaron las corridas incluidas (dólares):

| | Sol, calibrado | Sol, laxo | Luna, calibrado | Luna, laxo |
|---|---|---|---|---|
| Anuncios (578 pedazos) | 2.45 | 1.14 | 0.12 | 0.06 |
| Minutas (2,113 pedazos) | 4.37 | 4.44 | 0.23 | 0.23 |

---

## 6. Cómo se construye el índice

1. **Documentos.** Se descargan los PDFs de cada anuncio (238, 2000–2026) y cada minuta (126, 2011–2026) y se extrae el texto.
2. **Pedazos (chunks).** Cada documento se divide en pedazos de hasta ~3,000 caracteres, sin cortar párrafos. Un anuncio tiene ~2.4 pedazos; una minuta, ~17 (de 10 en 2011 a 23 hoy).
3. **Calificación.** Cada pedazo se le manda a un LLM con un prompt (en `src/index/llm_prompts.py`). El LLM responde:
   - `relevant`: si el pedazo habla de la economía o la política monetaria (sí/no);
   - `uncertainty_score`: de **−2** (mucha certeza) a **+2** (mucha incertidumbre);
   - `confidence` y `evidence` (la frase que justifica la calificación).
4. **Índice por documento** = el promedio de `uncertainty_score` en los pedazos marcados como relevantes.

### Especificación principal
Se probaron 2 prompts × 2 modelos = 4 versiones:
- **prompts**: *calibrado* (con ejemplos de qué es cada calificación) y *laxo* (sólo la definición);
- **modelos**: `openai/gpt-6.1-sol` (razonamiento mínimo) y `openai/gpt-6-luna` (más barato).

La versión principal, calibrado + Sol, se eligió antes de ver los resultados. No se escogió por cuál se parecía más al peso. Las 4 versiones se mueven juntas (correlación 0.70–0.93 suavizadas), así que el resultado no depende de esa elección. Todas las gráficas y los CSV usan la versión principal; el notebook 06 compara las cuatro.

### Los otros dos métodos (para comparar)

| Método | Idea | Resultado |
|---|---|---|
| **M1, diccionario** | Cuenta palabras de incertidumbre cerca de palabras de un tema, por cada 1,000 palabras | Las listas se ajustaron con los informes trimestrales y **no funcionan bien** en anuncios ni minutas |
| **M2, embeddings** | Convierte cada pedazo en un vector y mide qué tan cerca está de frases de incertidumbre menos frases de certeza | Funciona casi tan bien como el LLM |
| **M3, LLM** | Lo descrito arriba | El que se usa como índice principal |

Correlación con la volatilidad del peso:

| | anuncios, por documento | anuncios, suavizado | minutas, por documento | minutas, suavizado |
|---|---|---|---|---|
| M1 | 0.11 | 0.36 | 0.04 | −0.10 |
| M2 | 0.33 | 0.53 | 0.35 | 0.30 |
| M3 | 0.37 | **0.57** | 0.37 | 0.26 |

---

## 7. Frecuencia (un punto por decisión)

**Cada punto del índice es una decisión de política monetaria.** La frecuencia la pone el calendario de Banxico.

| Periodo | Decisiones por año |
|---|---|
| 2000–2002 | 3–6 |
| 2003–2010 | 11–14 |
| 2011 en adelante | 8 (calendario fijo, como la Reserva Federal) |

Entre una reunión y otra, Banxico no publica nada, así que no hay nada que medir. El índice mide lo que dice el banco, y el banco habla en sus reuniones, por eso el documento es la unidad.

Tambien se hicieron estas frecuencias.

- **Mensual**: repetir la última lectura hasta la siguiente decisión (como se hace con la tasa de política). Cada mes, el índice es lo último que dijo el banco, que es lo que el mercado conocía.
- **Trimestral**: el promedio de los documentos de cada trimestre. También está en `resultados/indice_trimestral.csv`. Sólo el cuarto trimestre de 2001 no tiene anuncio.

---

## 8. Promedio móvil

**Qué es**: el promedio de los últimos **8 documentos** (≈ 1 año desde 2011), hacia atrás. Se necesitan al menos 4 documentos para calcularlo.

Un anuncio tiene sólo 2–3 pedazos, así que cada punto es ruidoso. Un solo pedazo con una frase de riesgo puede mover mucho el punto. El promedio móvil deja ver la tendencia. Los puntos se muestran también para no esconder los datos.


---

## 9. Estandarización

```
indice_z = (indice − promedio) / desviación estándar
```

El promedio y la desviación estándar se calculan con toda la muestra de cada corpus, por separado:

| | promedio | desv. estándar |
|---|---|---|
| Anuncios, por decisión | 0.57 | 0.63 |
| Minutas, por decisión | 0.94 | 0.24 |
| Anuncios, trimestral | 0.56 | 0.56 |
| Minutas, trimestral | 0.94 | 0.20 |

0 = el documento típico de ese corpus; +1 = una desviación estándar más incierto de lo normal. 

Las minutas varían poco (desv. est. 0.24 contra 0.63 de los anuncios). Por eso estandarizadas, sus movimientos se ven más grandes.

---

## 10. CSV resultados

Sólo versión principal (calibrado + Sol). Una celda vacía significa que no hay documento (por ejemplo, minutas antes de 2011).

### `indice_por_decision.csv`: un renglón por decisión (238)

| Columna | Qué es |
|---|---|
| `fecha_decision` | fecha de la decisión |
| `anuncio_id` | id del anuncio |
| `anuncio_indice` | índice del anuncio, escala −2…+2 |
| `anuncio_promedio_movil` | promedio móvil de 8 anuncios |
| `anuncio_indice_z` | índice estandarizado |
| `anuncio_promedio_movil_z` | promedio móvil del índice estandarizado |
| `minuta_id`, `minuta_indice`, `minuta_promedio_movil`, `minuta_indice_z`, `minuta_promedio_movil_z` | lo mismo para la minuta de esa reunión |

### `indice_trimestral.csv`: un renglón por trimestre (107)

| Columna | Qué es |
|---|---|
| `fin_de_trimestre` | último día del trimestre |
| `trimestre` | p. ej. `2020Q2` |
| `anuncio_n_docs` | cuántos anuncios hubo en el trimestre |
| `anuncio_indice` | promedio de los anuncios del trimestre |
| `anuncio_suavizado_6t` | promedio móvil de 6 trimestres |
| `anuncio_indice_z`, `anuncio_suavizado_6t_z` | estandarizados (con promedio y desv. est. trimestrales) |
| `minuta_*` | lo mismo para las minutas |

---

## 11. Figuras 

La sección 11 de `06_resultados` guarda en `resultados/figuras/` las figuras relevantes. El nombre de cada archivo dice cuál es (p. ej. `01_indice_anuncios.png`, `09_indice_anuncios_z.png`).

---

## 12. Notas

- **Las minutas se hicieron más largas** (de ~10 a ~23 pedazos). Más texto rutinario baja el promedio, lo que probablemente explica parte de su caída desde 2023.
- **Episodios.** El índice capta 2008, COVID y la elección de EE. UU. de 2016; capta también la inflación de 2021–22, que el peso no refleja. No capta el taper de 2013 ni la elección mexicana de 2024 (Banxico evita hablar de política interna).
- Banxico adoptó el esquema de tasa objetivo en enero de 2008, sustituyendo formalmente el corto que operaba desde 1998. 
