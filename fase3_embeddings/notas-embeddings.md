# Embeddings

    ... → DataLoader → ids [batch, max_length] → **embeddings** → modelo

El DataLoader entrega ids (enteros). El modelo no opera con enteros sueltos:
necesita vectores. Los embeddings son la capa que convierte cada id en un
vector de números reales aprendible.

## Qué es un embedding

Una **tabla de consulta (lookup table)**: una matriz de pesos de tamaño
`[vocab_size, output_dim]`. Cada fila es el vector asociado a un id del
vocabulario. "Embeddear" un id = ir a esa fila y devolverla.

En PyTorch es `nn.Embedding(vocab_size, output_dim)`. Sus pesos se inicializan
al azar y se **aprenden durante el entrenamiento** como cualquier otro parámetro
(backprop). 

Equivale a multiplicar un one-hot del id por la matriz de pesos, pero se
implementa como lookup directo por eficiencia (no hace falta el one-hot).

## De id a vector: ejemplo

`vocab_size = 6`, `output_dim = 3` → matriz de pesos `6 × 3`.

    id = 3  →  fila 3  →  [ 0.21, -0.83, 0.05 ]   (valores aprendibles)

Con un lote real de ids `[8, 4]` (batch=8, max_length=4) y `output_dim = 256`:

    ids [8, 4]  →  token embeddings [8, 4, 256]

Cada id se ha sustituido por su vector de 256 dimensiones.

## Por qué embeddings posicionales

La autoatención **no distingue el orden**: el mismo token da el mismo vector
esté en la posición que esté. Sin más, el modelo no sabría distinguir
"perro muerde hombre" de "hombre muerde perro".

Solución de GPT: **embeddings posicionales absolutos**, otra tabla
`nn.Embedding(context_length, output_dim)`. Se embeddean las posiciones
`0, 1, ..., max_length-1`:

    posiciones [0..3]  →  positional embeddings [4, 256]

(GPT-2 usa posicionales absolutos y **aprendibles**; el transformer original
usaba sinusoidales fijos — aquí seguimos la versión de GPT.)

## Embedding final

La entrada real al modelo es la **suma** de ambos:

    input_embeddings = token_emb [8, 4, 256] + pos_emb [4, 256]

El posicional `[4, 256]` se suma por broadcasting a las 8 secuencias del lote.
Resultado: `[8, 4, 256]`, listo para entrar en los bloques transformer.