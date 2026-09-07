# Tokenizador — Notas

## 1. Qué es un tokenizador

Un modelo de lenguaje no entiende texto: opera sobre números. El tokenizador es
la pieza que traduce entre ambos mundos. Convierte una cadena de texto en una
secuencia de enteros (los *ids* o *tokens*) y viceversa. Es la frontera entre el
lenguaje humano y el espacio numérico en el que vive el modelo.

Expone dos operaciones simétricas:
- **encode**: texto → lista de ids
- **decode**: lista de ids → texto

## 2. Por qué es necesario

- Una red neuronal solo hace álgebra (multiplicaciones de matrices) sobre
  vectores de números reales. No puede "leer" caracteres.
- Cada id se usa después como índice en la **tabla de embeddings**, que asigna a
  cada token un vector aprendible. Sin ids no hay embeddings.
- El modelo necesita un **vocabulario finito y fijo**: su capa de salida tiene un
  tamaño igual al del vocabulario (predice una probabilidad por token posible).
  Por eso hay que decidir de antemano qué unidades existen.

## 3. Qué hace exactamente

**encode**:
1. Trocea el texto en unidades. Aquí, con una regex que separa por espacios y por
   signos de puntuación, tratando la puntuación como tokens independientes.
2. Limpia: quita espacios sobrantes y descarta cadenas vacías.
3. Mapea cada token a su id según el vocabulario. Si un token no está en el
   vocabulario, lo sustituye por el token especial `<|unk|>` (unknown).

**decode**:
1. Mapea cada id de vuelta a su string.
2. Los une con espacios.
3. Corrige el espaciado alrededor de la puntuación con una regex (elimina el
   espacio sobrante antes de `,` `.` `!` etc.).

> Nota: este `decode` **no es un inverso perfecto** del `encode`. El espaciado
> original se pierde y se reconstruye con reglas. Es una limitación conocida del
> enfoque a nivel de palabra.

## 4. Cómo lo construimos

**Paso previo — el vocabulario:**
- Partimos de un corpus de texto.
- Lo tokenizamos con la misma regex.
- Sacamos el conjunto de tokens *únicos*, los ordenamos y asignamos un id a cada
  uno (0, 1, 2, …).
- Añadimos tokens especiales: `<|unk|>` para lo desconocido y (más adelante)
  `<|endoftext|>` para separar documentos.
- El vocab es un diccionario `str → int`. Su inverso `int → str` lo construimos
  para poder decodificar.

**La clase (`SimpleTokenizerV1`):**
- `__init__` recibe el vocab (`str → int`) y construye el diccionario inverso
  (`int → str`).
- `encode`: usa `str → int` para pasar de tokens a ids.
- `decode`: usa `int → str` para pasar de ids a tokens.

## 5. Limitaciones de esta V1 

- **Vocabulario a nivel de palabra**: cualquier palabra no vista se colapsa a
  `<|unk|>` y se pierde información. Cubrir un idioma entero exigiría un
  vocabulario gigantesco.
- El espaciado se reconstruye con heurísticas, no se preserva.
- No maneja subpalabras, morfología, otros idiomas, emojis ni código.

**La solución de la industria: Byte Pair Encoding (BPE)**, tokenización a nivel de
*subpalabra*. Arranca de bytes/caracteres y va fusionando los pares más frecuentes
hasta alcanzar un tamaño de vocabulario objetivo. Ventajas: no necesita `<|unk|>`
(siempre puede descomponer hasta bytes), controla el tamaño del vocabulario
(~50k en GPT-2) y maneja palabras nuevas partiéndolas en trozos conocidos.

Por eso más adelante pasaremos a **tiktoken** (la implementación de BPE de OpenAI,
la que usan GPT-2/3/4). Pero construir esta V1 a mano es justo lo que nos permite
entender qué problema resuelve BPE.