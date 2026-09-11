# Attention — Notas

## Qué resuelve la atención
Cada token mira a todos los demás y se reconstruye como una media ponderada de
ellos (el *context vector*). El peso de cada token depende de cuánto "encaja"
con el actual. Eso es lo que deja que el modelo use contexto en vez de tokens
aislados.

## Scaled dot-product con Q/K/V (`selfAttentionv2.py`)
Aquí entran los pesos aprendibles. Cada token se proyecta a tres roles:
- **Query**: qué busco.
- **Key**: qué ofrezco para que me encuentren.
- **Value**: qué aporto si me eligen.


Ahora sí aprende: las tres matrices se ajustan por backprop.

Detalle:  `keys.T`. Funciona porque la entrada es 2D (un solo bloque). Con
batch (3D) `.T` deja de valer y necesitas `transpose(1, 2)` — justo lo que ya
haces en la causal.

## Causal attention + dropout (`causalAttention.py`)
Dos cosas nuevas sobre v2:
- **Máscara causal**: un token no puede mirar al futuro. Se pone a −inf la parte
  triangular superior de los scores *antes* del softmax, así esos pesos quedan a
  0. La máscara la guardas como buffer con `torch.triu(..., diagonal=1)`.
- **Dropout** sobre los pesos de atención, para regularizar.
- Ya trabaja en 3D `(b, num_tokens, d_in)` y usa `transpose(1, 2)`: correcto
  para batch.

