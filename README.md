# Práctica de Procesamiento de Lenguaje Natural
## HopeEXP

Se ha desarrollado un sistema de procesamiento del lenguaje natural de HopeEXP (https://www.codabench.org/competitions/13563/) una de las tareas pertenencientes a IberLEF 2026 (https://sites.google.com/view/iberlef-2026/tasks). 

HOPE-EXP tiene como objetivo ir más allá de la detección típica de la esperanza y proporcionar una comprensión estructurada a múltiples niveles de las expresiones de esperanza en las publicaciones de redes sociales.

Esta tarea está formada por 4 subtareas:
### Tarea A — Clasificación de Etiqueta Primaria (Etiqueta única)
Clasifica la publicación en:
- Esperanza Realista (Realistic Hope): Una declaración que expresa un deseo o expectativa de un resultado futuro que es plausible, alcanzable o basado en condiciones del mundo real, incluso si es incierto.
- Esperanza Irrealista (Unrealistic Hope): Una declaración que expresa un deseo o expectativa de un resultado futuro que es altamente improbable, imposible, fantástico o desconectado de las limitaciones reales.
- Esperanza General (General Hope): Una expresión vaga o amplia de optimismo o expectativa positiva sobre el futuro sin especificar un resultado concreto y claramente definido.
- Desesperanza (Hopelessness): Una declaración que indica la ausencia de esperanza, caracterizada por el pesimismo, la resignación o la creencia de que no ocurrirá ningún cambio positivo o resultado deseable.
- Esperanza Sarcástica (Sarcastic Hope): Una expresión irónica o burlona que superficialmente parece transmitir esperanza o expectativa, pero que en realidad implica crítica, incredulidad o el sentimiento opuesto.
- Sin Esperanza (Not Hope): Una declaración que no expresa ningún deseo, expectativa o anticipación de un resultado futuro, incluyendo descripciones neutras, afirmaciones fácticas, opiniones, emociones o narrativas sin una intención orientada al futuro.

### Tarea B — Detección de Emoción Desencadenante (Multietiqueta)
Identifica una o más emociones asociadas con la publicación:
- Tristeza, alegría, amor, ira, miedo, sorpresa, Neutral/no claro.

> Reglas: Se permite la selección de múltiples etiquetas.

### Tarea C — Extracción de Fragmentos (Identificación del Resultado)
Si la etiqueta primaria pertenece a cualquier categoría de Esperanza, los sistemas deben extraer hasta 3 fragmentos (spans) que describan el resultado esperado o evitado.
Cada fragmento debe:
- Ser una subcadena exacta del texto de entrada.
- Corresponder a un resultado deseado o evitado.
- Si la etiqueta primaria es Sin Esperanza o Desesperanza, la lista de anotaciones de fragmentos (span_annotations) debe estar vacía.

> La detección de fragmentos se evalúa utilizando: ROUGE 1.

Implementación inicial disponible en este repositorio:

- Entrenamiento supervisado BIO para spans extractivos en `train_task_c_spans.py`.
- Inferencia y generación de submission en `predict_task_c_spans.py`.
- Utilidades de alineación, reconstrucción y evaluación de spans en `src/span_utils.py`.

### Tarea D — Clasificación del Rol del Resultado (Condicional)
Para los fragmentos predichos que coincidan:
- Postura del resultado (outcome_stance): Deseado vs. Evitado.
- Actor: Uno mismo (Self), Otro (Other), Mundo/Sistema (World/System), No claro.

> Solo se calcula cuando los fragmentos están alineados.
