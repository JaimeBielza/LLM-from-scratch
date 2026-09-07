# DataLoader — Notas 





    texto crudo → tokenizador → ids → **DataLoader** → embeddings → modelo

El tokenizador convierte texto en ids. El DataLoader es el paso siguiente:
coge esa larga secuencia de ids y la organiza en **lotes de pares
(entrada, objetivo)** listos para entrenar. Es la pieza que decide *qué trozos*
de texto ve el modelo y *qué tiene que predecir* en cada uno.

## Para qué un Data Loader

Un LLM se entrena con una tarea muy concreta: **predecir el siguiente token**.
Para eso necesitamos, a partir del texto, generar ejemplos de la forma:

- entrada (input): una secuencia de tokens
- objetivo (target): esa misma secuencia **desplazada una posición a la derecha**

Es decir, el objetivo de cada posición es "el token que viene después". El
DataLoader es quien fabrica esos pares de forma masiva y eficiente.

## La ventana deslizante (sliding window)

La idea central. Dada la secuencia de ids del texto, se recorre con una ventana
de tamaño fijo `max_length` (el tamaño de contexto). Para cada ventana:

- `input`  = tokens[i : i + max_length]
- `target` = tokens[i + 1 : i + max_length + 1]   ← lo mismo, corrido +1

Ejemplo con `max_length = 4` y la secuencia `[1, 2, 3, 4, 5, 6, 7]`:

    input:  [1, 2, 3, 4]      target: [2, 3, 4, 5]
    input:  [2, 3, 4, 5]      target: [3, 4, 5, 6]
    ...


### El parámetro `stride`

`stride` es cuánto avanza la ventana entre un ejemplo y el siguiente:

- `stride = 1` → ventanas muy solapadas, máximo número de ejemplos (más
  aprovechamiento del texto, pero más redundancia).
- `stride = max_length` → ventanas sin solapamiento, cada token aparece una sola
  vez como entrada (menos ejemplos, sin repetición).



## La clase Dataset


- `__init__`: recibe el texto completo, lo tokeniza entero una vez, y con la
  ventana deslizante (`max_length`, `stride`) precalcula y guarda dos listas de
  tensores: los `input_ids` y los `target_ids` de todos los ejemplos.
- `__len__`: devuelve cuántos ejemplos hay (cuántas ventanas salieron).
- `__getitem__(idx)`: devuelve el par (input, target) número `idx`.



## El DataLoader de PyTorch

Encima del Dataset, se usa `torch.utils.data.DataLoader`, que es lo que añade
todo lo demás para entrenar en la práctica. 

