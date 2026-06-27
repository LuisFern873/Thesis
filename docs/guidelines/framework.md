# Guía para la Redacción del Capítulo II: Marco Teórico

## Objetivo del Capítulo

El marco teórico tiene como finalidad proporcionar el sustento conceptual y teórico que fundamenta la investigación. Debe presentar de manera estructurada los conceptos, principios, modelos, métodos y teorías necesarios para comprender el problema de investigación, interpretar los resultados y justificar las decisiones metodológicas adoptadas.

Este capítulo no busca describir investigaciones previas (eso corresponde al capítulo de Antecedentes), sino explicar los conceptos y fundamentos teóricos sobre los cuales se construye la investigación.

Al finalizar la lectura del capítulo, el lector debe comprender claramente:

* Los conceptos principales utilizados en la investigación.
* Las teorías y principios relevantes.
* Las relaciones entre los conceptos estudiados.
* Las definiciones empleadas para interpretar los resultados.
* El fundamento conceptual de la propuesta de investigación.

---

# Principios Generales de Redacción

## Estilo

El capítulo debe redactarse:

* En tiempo presente.
* Con estilo académico formal.
* De manera explicativa y conceptual.
* Con lenguaje técnico preciso.
* Manteniendo coherencia entre conceptos.
* Evitando discusiones de resultados experimentales.

---

## Diferencia respecto a los Antecedentes

### Antecedentes

Responden:

* ¿Qué han investigado otros autores?
* ¿Qué resultados han obtenido?
* ¿Qué vacíos existen?

### Marco Teórico

Responde:

* ¿Qué conceptos explican el fenómeno estudiado?
* ¿Qué teorías sustentan la investigación?
* ¿Cómo se relacionan los elementos analizados?
* ¿Qué definiciones serán utilizadas durante el estudio?

---

# Principio de Construcción

El marco teórico debe derivarse directamente de:

* El título de la investigación.
* El problema de investigación.
* El objetivo general.
* Los objetivos específicos.
* Las palabras clave del estudio.

Todo concepto incluido debe justificar claramente su relación con la investigación.

---

# Organización del Capítulo

## Estructura General

El contenido debe organizarse mediante una estructura jerárquica coherente.

### Nivel 1

Corresponde a los grandes temas de la investigación.

### Nivel 2

Corresponde a conceptos específicos dentro de cada tema.

### Nivel 3

Corresponde a detalles técnicos o clasificaciones particulares.

### Restricción

No utilizar más de tres niveles de profundidad.

Ejemplo:

```text
2.1 Aprendizaje Federado

    2.1.1 Arquitectura General

    2.1.2 Desafíos del Aprendizaje Federado

        2.1.2.1 Heterogeneidad Estadística

2.2 Arquitecturas de Visión

    2.2.1 CNN

    2.2.2 Vision Transformers

    2.2.3 Vision Mamba
```

---

# Contenido Esperado

## 1. Definiciones Fundamentales

### Objetivo

Introducir los conceptos esenciales del dominio de investigación.

### Debe incluir

* Definiciones ampliamente aceptadas.
* Terminología especializada.
* Conceptos necesarios para comprender el problema.

### Evitar

* Definiciones de diccionario.
* Definiciones sin respaldo académico.
* Conceptos irrelevantes para la investigación.

---

## 2. Fundamentos Teóricos

### Objetivo

Explicar los principios que sustentan el funcionamiento de los fenómenos estudiados.

### Debe incluir

* Modelos teóricos.
* Principios matemáticos.
* Mecanismos de funcionamiento.
* Relaciones entre variables o componentes.

### Preguntas que debe responder

* ¿Cómo funciona el fenómeno estudiado?
* ¿Por qué ocurre?
* ¿Qué principios lo explican?

---

## 3. Clasificaciones y Taxonomías

### Objetivo

Organizar los conceptos según categorías reconocidas en la literatura.

### Debe incluir

* Clasificaciones relevantes.
* Comparaciones conceptuales.
* Características distintivas.

### Ejemplo conceptual

Si el tema es aprendizaje federado:

* Aprendizaje centralizado.
* Aprendizaje distribuido.
* Aprendizaje federado horizontal.
* Aprendizaje federado vertical.
* Aprendizaje federado híbrido.

---

## 4. Relación entre Conceptos

### Objetivo

Explicar cómo interactúan los elementos involucrados en la investigación.

### Debe responder

* ¿Qué dependencia existe entre los conceptos?
* ¿Cómo afecta un fenómeno a otro?
* ¿Qué elementos intervienen en el problema estudiado?

Esta sección es especialmente importante porque conecta el marco teórico con los objetivos de investigación.

---

## 5. Conceptos Asociados a la Metodología

### Objetivo

Fundamentar las herramientas y métodos que serán utilizados posteriormente.

### Debe incluir

* Modelos matemáticos relevantes.
* Métricas utilizadas.
* Algoritmos.
* Técnicas de análisis.
* Métodos de evaluación.

Solo deben incluirse aquellos conceptos que realmente serán utilizados en la investigación.

---

# Recomendaciones para una Investigación sobre Federated Learning y Arquitecturas de Visión

El marco teórico debería cubrir únicamente los conceptos necesarios para comprender el problema y la metodología.

## Posible estructura

### 2.1 Aprendizaje Federado

* Definición.
* Arquitectura cliente-servidor.
* Proceso de entrenamiento federado.
* Ventajas y desafíos.

### 2.2 Heterogeneidad de Datos

* Datos IID y Non-IID.
* Tipos de heterogeneidad.
* Impacto sobre el aprendizaje federado.

### 2.3 Algoritmos de Optimización Federada

* FedAvg.
* FedProx.
* SCAFFOLD.
* Otros métodos relevantes utilizados en la investigación.

### 2.4 Arquitecturas de Visión Computacional

#### CNN

* Principios básicos.
* Capas convolucionales.
* Extracción jerárquica de características.

#### Vision Transformer (ViT)

* Tokenización.
* Autoatención.
* Representaciones globales.

#### Vision Mamba

* State Space Models.
* Procesamiento secuencial eficiente.
* Diferencias respecto a Transformers.

#### Vision GNN

* Representación basada en grafos.
* Construcción de relaciones espaciales.
* Propagación de mensajes.

### 2.5 Análisis de Representaciones

* Similaridad de representaciones.
* Centered Kernel Alignment (CKA).
* Drift de representaciones.
* Interferencia entre componentes del modelo.

### 2.6 Métricas de Evaluación

* Accuracy.
* Convergencia.
* Fairness.
* Robustez.
* Métricas específicas utilizadas en el estudio.

---

# Qué Debe Evitar el Agente

No incluir:

❌ Revisiones extensas de artículos.

❌ Resultados experimentales de trabajos previos.

❌ Comparaciones bibliográficas detalladas.

❌ Información histórica innecesaria.

❌ Conceptos no utilizados en la investigación.

❌ Explicaciones excesivamente matemáticas que no contribuyan a los objetivos.

❌ Secciones añadidas únicamente para aumentar la extensión.

---

# Criterios de Calidad

Un buen marco teórico debe:

* Derivarse directamente del título y objetivos.
* Contener únicamente conceptos relevantes.
* Mantener coherencia entre secciones.
* Facilitar la comprensión de la metodología.
* Servir como base para interpretar los resultados.
* Definir claramente toda terminología especializada utilizada posteriormente.

Cada sección debe responder implícitamente a la pregunta:

> "¿Qué necesita comprender el lector para entender correctamente la investigación y los resultados que se presentarán más adelante?"

Si un concepto no ayuda a responder esa pregunta, probablemente no debe formar parte del marco teórico.

---

# Principio Fundamental

El marco teórico no debe explicar qué hicieron otros investigadores; debe explicar qué conceptos, teorías, principios y modelos permiten comprender el problema, la metodología y los resultados de la investigación.

Todo contenido debe contribuir a construir el lenguaje conceptual común entre el investigador y el lector que será utilizado en el resto de la tesis.
