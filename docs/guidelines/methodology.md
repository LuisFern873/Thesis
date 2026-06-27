# Guía para la Redacción del Capítulo: Marco Metodológico

## Objetivo del Capítulo

El marco metodológico tiene como finalidad describir de manera precisa cómo se realizó la investigación y cómo se alcanzaron los objetivos planteados.

Debe responder explícitamente las siguientes preguntas:

* ¿Cómo se lograron los objetivos de investigación?
* ¿Qué procedimientos se siguieron?
* ¿Qué métodos, técnicas, herramientas e instrumentos se utilizaron?
* ¿Cómo se obtuvieron y analizaron los datos?
* ¿Cómo puede otro investigador reproducir el estudio?

El capítulo constituye el puente entre la Introducción (lo que se propuso hacer) y los Resultados (lo que se obtuvo).

---

# Principios Generales de Redacción

## Estilo

El capítulo debe redactarse:

* En tiempo pasado.
* Con estilo informativo y declarativo.
* De forma objetiva.
* Priorizando la descripción de procedimientos.
* Sin explicaciones teóricas extensas.

---

## Principio Fundamental

El lector debe ser capaz de reproducir completamente la investigación utilizando únicamente la información contenida en este capítulo.

Todo procedimiento relevante debe quedar explícitamente documentado.

---

# Estructura General

El capítulo debe organizarse en tres grandes secciones:

1. Caracterización de la investigación.
2. Descripción del diseño de investigación.
3. Consideraciones éticas.

La exposición debe avanzar desde aspectos generales hacia detalles específicos.

---

# 1. Caracterización de la Investigación

## Objetivo

Clasificar formalmente la investigación según criterios metodológicos reconocidos.

---

## 1.1 Según la Utilidad del Resultado

### Investigación Aplicada

Se utiliza cuando el objetivo es resolver un problema concreto o generar una solución práctica.

### Investigación Fundamental

Se utiliza cuando el objetivo principal es generar nuevo conocimiento teórico.

### Recomendación

Para investigaciones en aprendizaje federado, arquitecturas de visión y evaluación experimental, normalmente corresponde:

> Investigación aplicada.

---

## 1.2 Según el Enfoque

### Cuantitativo

Cuando los resultados se obtienen mediante mediciones, métricas y análisis numéricos.

### Cualitativo

Cuando predominan interpretaciones, percepciones o análisis descriptivos.

### Recomendación

Para estudios experimentales de machine learning normalmente corresponde:

> Enfoque cuantitativo.

---

## 1.3 Según el Diseño

### Experimental

Cuando el investigador manipula variables y evalúa sus efectos.

### No Experimental

Cuando únicamente observa fenómenos existentes sin intervención directa.

### Recomendación

En investigaciones que comparan modelos, arquitecturas o configuraciones experimentales normalmente corresponde:

> Diseño experimental.

---

## 1.4 Según la Temporalidad

### Transversal

Los datos se recolectan en un único período de observación.

### Longitudinal

Los fenómenos se observan a lo largo del tiempo.

### Recomendación

La mayoría de estudios experimentales en aprendizaje automático corresponden a:

> Diseño longitudinal.

Debido a que se analiza la evolución del entrenamiento a través de múltiples rondas o épocas.

---

# 2. Descripción del Diseño de Investigación

## Objetivo

Explicar detalladamente cómo se ejecutó la investigación.

Esta sección debe organizarse siguiendo la lógica de los objetivos específicos.

---

# Principio de Organización

Cada objetivo específico debe convertirse en una subsección metodológica.

Para cada objetivo específico se debe responder:

* ¿Qué se hizo?
* ¿Cómo se hizo?
* ¿Con qué herramientas se hizo?
* ¿Qué datos se utilizaron?
* ¿Qué resultado produjo?

---

# Estructura Recomendada por Objetivo

## Objetivo Específico X

### Actividades realizadas

Describir las actividades necesarias para alcanzar el objetivo.

### Método utilizado

Indicar el método empleado.

### Técnicas aplicadas

Describir las técnicas específicas utilizadas.

### Instrumentos

Especificar instrumentos o mecanismos de medición.

### Herramientas

Indicar software, bibliotecas, plataformas o infraestructura utilizada.

### Resultado esperado

Describir el producto o evidencia generada por la actividad.

---

# Elementos Metodológicos que Deben Describirse

## Datos Utilizados

### Debe incluir

* Dataset utilizado.
* Fuente del dataset.
* Número de muestras.
* Número de clases.
* Preprocesamiento aplicado.
* División entrenamiento/prueba.

### Ejemplo

* CIFAR-10.
* Brain Tumor MRI.
* MedMNIST.
* Otros datasets utilizados.

---

## Configuración Experimental

### Debe incluir

* Número de clientes.
* Escenarios IID y Non-IID.
* Valores de heterogeneidad (α).
* Número de rondas federadas.
* Tamaño de lote.
* Learning rate.
* Optimizadores utilizados.

---

## Modelos Evaluados

### Debe incluir

* Arquitecturas comparadas.
* Configuración de cada arquitectura.
* Cantidad de parámetros.
* Adaptaciones realizadas.

Ejemplo:

* CNN.
* Vision Transformer.
* Vision Mamba.
* Vision GNN.

---

## Algoritmos Federados

### Debe incluir

* FedAvg.
* FedProx.
* SCAFFOLD.
* Otros métodos evaluados.

Especificar parámetros relevantes de cada algoritmo.

---

## Variables del Estudio

### Variables Independientes

Factores manipulados por el investigador.

Ejemplos:

* Arquitectura utilizada.
* Nivel de heterogeneidad.
* Algoritmo federado.

### Variables Dependientes

Resultados medidos.

Ejemplos:

* Accuracy.
* Convergencia.
* Fairness.
* Drift.
* Similitud CKA.

### Variables Intervinientes

Factores controlados durante el experimento.

Ejemplos:

* Dataset.
* Número de rondas.
* Hardware utilizado.

---

## Métricas de Evaluación

Debe describirse:

* Qué métricas fueron utilizadas.
* Cómo fueron calculadas.
* Qué aspecto del desempeño representan.

Ejemplos:

* Accuracy.
* F1-score.
* Convergence Round.
* Fairness.
* Raw Drift.
* Normalized Drift.
* Interference.
* CKA Similarity.

No es necesario redefinir matemáticamente métricas estándar ampliamente conocidas.

---

## Procedimiento Experimental

### Debe incluir

La secuencia completa de ejecución.

Ejemplo:

1. Preparación de datasets.
2. Generación de particiones federadas.
3. Configuración de escenarios IID y Non-IID.
4. Inicialización de modelos.
5. Entrenamiento federado.
6. Registro de métricas.
7. Cálculo de indicadores derivados.
8. Análisis estadístico.
9. Generación de resultados.

El procedimiento debe permitir reproducir exactamente los experimentos.

---

## Infraestructura Tecnológica

Debe especificarse:

### Hardware

* CPU.
* GPU.
* Memoria RAM.

### Software

* Sistema operativo.
* Python.
* PyTorch.
* Frameworks utilizados.

### Repositorios

* Código fuente.
* Configuraciones experimentales.

---

# Qué No Debe Incluirse

No incluir:

❌ Explicaciones conceptuales extensas de métodos ya descritos en el Marco Teórico.

❌ Revisiones bibliográficas.

❌ Resultados experimentales.

❌ Interpretaciones de resultados.

❌ Discusiones.

❌ Justificaciones teóricas extensas.

El foco debe estar exclusivamente en describir cómo se ejecutó la investigación.

---

# 3. Consideraciones Éticas

## Objetivo

Declarar explícitamente los principios éticos observados durante la investigación.

---

## 3.1 Confidencialidad

Describir las medidas adoptadas para proteger información sensible o privada.

Si se utilizaron datasets públicos y anonimizados, indicarlo explícitamente.

---

## 3.2 Propiedad Intelectual

Indicar que:

* Se respetaron licencias de software.
* Se citaron adecuadamente las fuentes utilizadas.
* Se reconoció la autoría de datasets, modelos y herramientas.

---

## 3.3 Honestidad Científica

Declarar que:

* No se manipularon resultados.
* No se fabricaron datos.
* Los resultados se reportaron de manera íntegra.

---

## 3.4 Objetividad

Indicar que:

* Las conclusiones se derivaron de evidencia empírica.
* Los resultados fueron evaluados mediante métricas objetivas.
* La aceptación o rechazo de hipótesis se basó en datos observables.

---

# Recomendaciones para una Investigación sobre Federated Learning

La estructura metodológica suele alinearse naturalmente con los objetivos específicos:

### Objetivo 1

Diseño de escenarios federados y heterogeneidad.

### Objetivo 2

Implementación y entrenamiento de arquitecturas.

### Objetivo 3

Evaluación cuantitativa del desempeño.

### Objetivo 4

Análisis de representaciones internas.

### Objetivo 5

Comparación e interpretación de resultados.

Cada objetivo debe transformarse en una descripción clara y reproducible de actividades.

---

# Criterios de Calidad

Un buen marco metodológico debe permitir responder:

* ¿Qué se hizo?
* ¿Cómo se hizo?
* ¿Con qué se hizo?
* ¿Qué se midió?
* ¿Cómo se analizaron los datos?
* ¿Cómo podría repetirse exactamente el estudio?

Si un investigador independiente puede replicar los experimentos únicamente leyendo este capítulo, entonces el marco metodológico cumple su propósito.

---

# Principio Fundamental

El marco metodológico no debe explicar por qué una técnica existe ni discutir sus ventajas teóricas; debe documentar con precisión cómo fue utilizada dentro de la investigación.

Su propósito principal es garantizar transparencia, trazabilidad y reproducibilidad científica.
