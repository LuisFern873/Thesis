# Guía para la Redacción del Capítulo V: Resultados y Discusión

## Objetivo del Capítulo

El capítulo tiene como finalidad presentar los resultados obtenidos durante la investigación y analizar su significado en relación con los objetivos planteados, el problema de investigación, las hipótesis (si existen) y el estado del arte.

Debe responder simultáneamente las preguntas:

* ¿Qué se encontró durante la investigación?
* ¿Qué significan los resultados obtenidos?
* ¿Cómo se relacionan con los objetivos planteados?
* ¿Cómo se comparan con investigaciones previas?
* ¿Qué aportes generan al conocimiento existente?

Este capítulo constituye el vínculo entre la metodología aplicada y las conclusiones finales.

---

# Principios Generales de Redacción

## Resultados

Los resultados deben:

* Presentarse de manera objetiva.
* Basarse en evidencias cuantitativas o cualitativas.
* Incluir tablas, figuras y gráficos cuando sea pertinente.
* Mantener el orden de los objetivos específicos.
* Reportar tanto resultados esperados como inesperados.

## Discusión

La discusión debe:

* Interpretar los resultados obtenidos.
* Explicar posibles causas.
* Contrastar los hallazgos con el estado del arte.
* Analizar implicancias metodológicas y científicas.
* Relacionar los hallazgos con los objetivos de investigación.

---

# Estilo de Redacción

## Sección de Resultados

* Tiempo pasado.
* Estilo expositivo y descriptivo.
* Basado en evidencias.
* Sin opiniones personales.

Ejemplo:

> Se observó una reducción de la precisión conforme aumentó el nivel de heterogeneidad estadística.

## Sección de Discusión

* Tiempo presente.
* Estilo analítico y argumentativo.
* Basado en evidencia empírica.

Ejemplo:

> Este comportamiento sugiere que las arquitecturas basadas en atención presentan una mayor sensibilidad a la heterogeneidad estadística.

---

# Estructura General

El capítulo debe organizarse siguiendo exactamente la lógica de los objetivos específicos.

## Principio Fundamental

Cada objetivo específico debe tener:

1. Presentación de resultados.
2. Discusión de resultados.

No presentar primero todos los resultados y luego toda la discusión.

La estructura recomendada es:

```text
5.1 Resultados y discusión del Objetivo Específico 1

    Resultados

    Discusión

5.2 Resultados y discusión del Objetivo Específico 2

    Resultados

    Discusión

5.3 Resultados y discusión del Objetivo Específico 3

    Resultados

    Discusión
```

---

# Estructura Recomendada por Objetivo

## 5.X Objetivo Específico X

### Presentación de Resultados

#### Objetivo

Mostrar evidencia obtenida durante la ejecución del estudio.

#### Debe incluir

* Tablas.
* Figuras.
* Métricas.
* Resultados experimentales.
* Estadísticas descriptivas.

#### Reglas

Presentar:

* Qué se midió.
* Qué valores se obtuvieron.
* Qué comportamiento se observó.

Sin interpretar.

### Discusión

#### Objetivo

Explicar el significado de los resultados.

#### Debe responder

* ¿Por qué ocurrió este resultado?
* ¿Es consistente con la literatura?
* ¿Contradice investigaciones previas?
* ¿Qué implicaciones tiene?
* ¿Qué limitaciones revela?

---

# Presentación de Tablas y Figuras

## Recomendaciones

Cada figura o tabla debe:

* Estar numerada.
* Tener título descriptivo.
* Ser referenciada desde el texto.
* Ser explicada antes o después de su aparición.

Ejemplo:

> La Figura 5.3 muestra la evolución de la precisión global durante las rondas federadas.

Evitar:

❌ Insertar gráficos sin explicación.

❌ Repetir textualmente todos los valores de una tabla.

❌ Describir información evidente visualmente.

---

# Tipos de Resultados Esperados en una Investigación de Federated Learning

## Desempeño Predictivo

Ejemplos:

* Accuracy.
* F1-score.
* Precision.
* Recall.

### Discusión

* Robustez.
* Generalización.
* Sensibilidad a heterogeneidad.

---

## Convergencia

Ejemplos:

* Accuracy por ronda.
* Curvas de entrenamiento.
* Convergence Round.

### Discusión

* Velocidad de convergencia.
* Estabilidad.
* Eficiencia de entrenamiento.

---

## Fairness

Ejemplos:

* Variabilidad entre clientes.
* Distribución del desempeño local.

### Discusión

* Equidad del aprendizaje.
* Impacto de Non-IID.

---

## Drift de Representaciones

Ejemplos:

* Raw Drift.
* Normalized Drift.

### Discusión

* Estabilidad representacional.
* Adaptación local.
* Especialización de capas.

---

## Similaridad de Representaciones

Ejemplos:

* CKA intra-round.
* CKA inter-round.

### Discusión

* Conservación de conocimiento.
* Transferencia de representaciones.
* Evolución del aprendizaje.

---

## Interferencia

Ejemplos:

* Interferencia entre capas.
* Interferencia entre componentes.

### Discusión

* Competencia representacional.
* Robustez arquitectónica.
* Efectos de la heterogeneidad.

---

# Cómo Construir la Discusión

La discusión debe seguir la siguiente secuencia:

## Paso 1

Recordar brevemente el resultado principal.

## Paso 2

Interpretar el comportamiento observado.

## Paso 3

Comparar con antecedentes relevantes.

## Paso 4

Explicar coincidencias o discrepancias.

## Paso 5

Indicar implicancias para la investigación.

---

# Comparación con el Estado del Arte

Cada discusión debe incluir, cuando sea posible:

### Coincidencias

Ejemplo:

> Este resultado coincide con estudios previos que reportan una degradación progresiva del desempeño bajo escenarios altamente Non-IID.

### Diferencias

Ejemplo:

> A diferencia de investigaciones anteriores, la arquitectura Vision Mamba mostró una menor sensibilidad al drift representacional.

### Aportes

Ejemplo:

> Este hallazgo aporta evidencia sobre el comportamiento interno de las representaciones en entornos federados heterogéneos.

---

# Qué Debe Evitar el Agente

No incluir:

❌ Conclusiones finales del estudio.

❌ Recomendaciones.

❌ Resúmenes extensos del marco teórico.

❌ Opiniones sin evidencia.

❌ Interpretaciones no respaldadas por resultados.

❌ Comparaciones con trabajos que no sean relevantes.

❌ Discusiones desconectadas de los objetivos.

---

# Criterios de Calidad

Un buen capítulo de Resultados y Discusión debe permitir responder:

* ¿Qué se encontró?
* ¿Qué evidencia lo demuestra?
* ¿Por qué ocurrió?
* ¿Cómo se relaciona con la literatura?
* ¿Qué aporta al conocimiento existente?

Cada objetivo específico debe quedar claramente respondido mediante evidencia y análisis.

---

# Principio Fundamental

Los resultados muestran la evidencia.

La discusión explica el significado de esa evidencia.

Cada objetivo específico debe quedar resuelto mediante una combinación explícita de ambos elementos, manteniendo una conexión directa con la metodología aplicada y con el estado del arte revisado previamente.
