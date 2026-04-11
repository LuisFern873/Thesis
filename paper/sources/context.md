
# Asunto: Avance de tesis y solicitud de orientación – PFC II

## Correo Luis Méndez 

Buenos días, profesora Aurea:

Mi nombre es Luis Méndez y soy estudiante del curso de PFC II. En PFC I mi asesor fue el profesor Víctor Flores; sin embargo, al momento de realizar mi matrícula no figuraba su nombre.

Quisiera compartirle el avance que desarrollé durante el curso de PFC I y solicitarle su orientación para continuar con el proyecto. Adjunto mi presentación final de PFC I, donde expuse los experimentos preliminares. De forma resumida, mi tema de investigación aborda el Aprendizaje Federado; los detalles completos se encuentran en la presentación.

Al finalizar el curso quedó pendiente continuar con los experimentos utilizando otros algoritmos de agregación. Hasta el momento solo he trabajado con el algoritmo más simple (FedAvg). Por ello, me gustaría consultarle si considera conveniente que continúe con nuevos experimentos o si sería mejor detenerme primero para reevaluar el alcance de la tesis (hipótesis, datasets y contribución).

Es importante mencionar que mis resultados preliminares contradicen la hipótesis planteada. Sospecho que esto podría estar relacionado con la diferencia en la cantidad de parámetros entre los modelos, aunque aún no estoy completamente seguro.

Asimismo, le comparto un análisis detallado de la presentación (generado con apoyo de IA), donde se describe el estado actual de la tesis, los problemas críticos identificados y algunos posibles caminos de mejora:
https://claude.ai/share/2f81449e-75bc-4e97-803a-9a8525d78c77

Mi objetivo al culminar el trabajo es poder presentar los resultados en formato de paper.

Quedo atento a sus comentarios profesora Aurea

Saludos cordiales,
Luis Méndez


## Correo respuesta de mi asesora Aurea Soriano-Vargas

Hola Luis, 

¿Todo bien?
He revisado en detalle tus avances y el análisis reciente de tu presentación. En general, el proyecto está bien encaminado y tienes ya una base experimental sólida, pero para llevarlo a un nivel adecuado de tesis de pregrado en el tiempo disponible, es importante que enfoques y cierres correctamente el trabajo.
Primero, respecto a la hipótesis inicial (“DeiT y Mamba superarán a CNNs”), es importante que la ajustes. La literatura sugiere que modelos como DeiT o Mamba pueden ser más robustos en ciertos escenarios (por su capacidad de modelar contexto global), pero no de forma consistente en todos los datasets ni configuraciones de Federated Learning. Tus propios resultados van en esa línea: las CNNs (especialmente EfficientNet y ConvNeXt) siguen siendo muy competitivas, e incluso superiores en varios casos. Por lo tanto, lo correcto sería algo como: DeiT y Mamba pueden superar a CNNs en algunos escenarios con heterogeneidad y cuando el modelado de contexto global importa, pero no de forma consistente en todos los datasets, tareas médicas o configuraciones de FL.

Alguna evidencia sobre ello:
1) En federated learning con datos no-IID, sí hay trabajos que sugieren que la arquitectura importa mucho y que los transformers pueden ganar a CNNs bajo alta heterogeneidad, especialmente en benchmarks de visión con particiones muy sesgadas; esto apoya la parte “DeiT > CNNs” como hipótesis plausible, al menos en ciertos escenarios (Himeur et al., 2023; He et al., 2021).
2) En imágenes médicas fuera de FL, Mamba y los modelos con contexto global suelen rendir muy bien, pero normalmente como “competitivos” o “superiores en varios datasets”, no como ganadores universales; por ejemplo, MedMamba reporta rendimiento competitivo en 16 datasets médicos, lo que respalda que Mamba es una opción seria, pero no prueba que siempre supere a CNNs (Yue & Li, 2024; Liu et al., 2024; Rahman et al., 2024).
3) También hay evidencia reciente en medicina de que Mamba puede superar a transformers y CNNs en tareas concretas, como clasificación histopatológica de Gleason; eso da apoyo a la idea “Mamba > CNNs” en dominios específicos, pero sigue siendo evidencia puntual y no una regla general (Mohammadi et al., 2024).
4) Sin embargo, la literatura médica reciente también muestra claramente que CNNs siguen siendo muy fuertes y a veces superan a transformers y Mamba, incluso bajo comparaciones directas; en rayos X de tórax, EfficientNet quedó entre los mejores modelos, y en segmentación dental varios CNNs superaron a transformers y Mamba (Yanar et al., 2025; Ghimire et al., 2025).
5) En FL médico, la tendencia general de los benchmarks no es que exista una arquitectura o algoritmo ganador universal, sino que el rendimiento cambia bastante según dataset, partición, heterogeneidad y costo de comunicación; eso va en contra de una hipótesis absoluta del tipo “DeiT y Mamba > CNNs” (Zhou et al., 2025; Shenaj et al., 2023).
Ahora, sobre el trabajo experimental, estas son las prioridades claras que debes seguir en las próximas semanas:

Consistencia y rigor experimental
Debes rehacer o extender los experimentos para que todos tengan:

100 communication rounds (no 20)

Al menos 3 corridas por configuración (con diferentes seeds)

Reporte de media y desviación estándar

Esto es crítico para que los resultados sean válidos.

Reducción del scope
Actualmente el espacio experimental es demasiado grande. Debes reducirlo a algo manejable:

2–3 modelos (por ejemplo: EfficientNet, DeiT y opcionalmente uno adicional)

1–2 datasets (priorizando el dataset médico)

2 métodos principales (FedAvg + 1 adicional FedProx o MOON)

El objetivo no es probar todo, sino analizar bien.

Métodos de Federated Learning
Aunque mencionas varios métodos (FedProx, MOON, etc.), en la práctica solo has evaluado FedAvg.
Debes implementar al menos uno adicional (recomendado: FedProx) y compararlo con FedAvg en un subset reducido de experimentos.

Análisis profundo (esto es lo más importante)
Actualmente describes resultados, pero no los explicas. Necesitas responder preguntas como:

¿Por qué las CNNs superan a DeiT en ciertos datasets?

¿Por qué Mamba/Vim falla consistentemente?

¿Cómo afecta el tipo de heterogeneidad a cada arquitectura?

Aquí está el verdadero aporte de tu tesis.

Análisis por cliente (FL-specific)
Debes incluir:

Performance por cliente

Variabilidad entre clientes

Identificar si hay clientes que sufren más (fairness)

Esto es clave en Federated Learning.

Metodología completa y reproducible
Debes documentar claramente:

Learning rate, optimizer, batch size

Número de clientes, epochs locales

Estrategia de partición (IID vs non-IID, cómo se genera)

Seeds utilizadas

Sección de Discussion
Debes escribir una discusión sólida donde:

Expliques por qué la hipótesis no se cumple completamente

Relacionas tus resultados con la literatura

Extraigas conclusiones prácticas (cuándo usar CNN vs Transformer, etc.)

Limpieza conceptual del trabajo
Tienes dos opciones válidas:

Reformular como “comparative study” (recomendado)

O mantener hipótesis, pero condicionada (más difícil)

Mi recomendación es la primera.

En resumen, ya tienes la parte técnica avanzada. Lo que falta ahora no es “más modelos”, sino:

consistencia experimental

reducción del scope

análisis profundo

Si haces estos ajustes, tu trabajo puede quedar muy sólido e incluso con potencial de publicación.

Podemos continuar con esta propuesta para overleaf

Saludos

--

Aurea