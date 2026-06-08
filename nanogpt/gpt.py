
import torch
import torch.nn as nn
from torch.nn import functional as F

# ===================== Hiperparámetros =====================
# batch_size: cuántas secuencias independientes procesamos en paralelo en cada paso
batch_size = 64 # how many independent sequences will we process in parallel?
# block_size: longitud máxima de contexto que el modelo puede ver para predecir el siguiente carácter
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # número total de pasos de entrenamiento
eval_interval = 500 # cada cuántos pasos medimos la pérdida en train/val
eval_iters = 200 # cuántos batches promediamos para estimar la pérdida


n_embd = 384 # dimensión de los embeddings (y del modelo en general)
n_head = 6 # número de cabezas de atención por bloque transformer
n_layer = 6 # número de bloques transformer apilados

learning_rate = 3e-4 # tasa de aprendizaje del optimizador AdamW
dropout = 0.2 # probabilidad de dropout, ayuda a regularizar y evitar overfitting
device = 'cuda' if torch.cuda.is_available() else 'cpu' # usa GPU si está disponible
# ------------

torch.manual_seed(1337) # semilla fija para que los resultados sean reproducibles

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
# ===================== Tokenización a nivel de carácter =====================
chars = sorted(list(set(text))) # vocabulario: todos los caracteres únicos del texto
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # string -> índice
itos = { i:ch for i,ch in enumerate(chars) } # índice -> string
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # todo el texto codificado como enteros
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # posiciones iniciales aleatorias para cada secuencia del batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # entradas: bloques de "block_size" caracteres
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # objetivos: el mismo bloque desplazado un carácter a la derecha
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # no necesitamos gradientes solo para evaluar, así ahorramos memoria y cómputo
def estimate_loss():
    # promedia la pérdida sobre varios batches para obtener una estimación menos ruidosa
    out = {}
    model.eval() # modo evaluación (desactiva dropout, etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # vuelve a modo entrenamiento (reactiva dropout, etc.)
    return out

# ===================== Bloques del Transformer =====================
class Head(nn.Module):
    """ one head of self-attention """
    # Una sola "cabeza" de self-attention: aprende a decidir, para cada token,
    # cuánta atención prestarle a los tokens anteriores (incluido él mismo).

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # proyecta x a "qué información ofrezco"
        self.query = nn.Linear(n_embd, head_size, bias=False) # proyecta x a "qué información busco"
        self.value = nn.Linear(n_embd, head_size, bias=False) # proyecta x a "qué información comparto si me prestan atención"
        # tril: máscara triangular inferior que impide que un token "vea" tokens futuros (atención causal)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        # producto punto query·key escalado por sqrt(head_size) para mantener varianzas controladas
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # bloquea la atención hacia el futuro -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # convierte los puntajes en pesos que suman 1 (B, T, T)
        wei = self.dropout(wei) # apaga aleatoriamente algunas conexiones de atención (regularización)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # combina los "values" según los pesos de atención -> (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    # Corre varias cabezas de atención en paralelo y junta sus resultados;
    # cada cabeza puede aprender a fijarse en relaciones distintas entre tokens.

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) # vuelve a proyectar al tamaño original del embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatena las salidas de todas las cabezas
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    # Red feed-forward posicional: procesa cada token de forma independiente,
    # dándole al modelo capacidad de "pensar" sobre la información ya recolectada por la atención.

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # expande la dimensión (factor 4, como en el paper original de Transformers)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # vuelve a comprimir a la dimensión original
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    # "Comunicación" = self-attention (los tokens intercambian información entre sí)
    # "Cómputo" = feed-forward (cada token procesa esa información por su cuenta)
    # Las conexiones residuales (x + ...) y las layer norms ayudan a entrenar redes profundas de forma estable.

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # cada cabeza recibe una porción de la dimensión total
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # normaliza antes de la atención (pre-norm)
        self.ln2 = nn.LayerNorm(n_embd) # normaliza antes del feed-forward (pre-norm)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # conexión residual: x + atención(norm(x))
        x = x + self.ffwd(self.ln2(x)) # conexión residual: x + feedforward(norm(x))
        return x

# ===================== Modelo GPT completo =====================
class GPTLanguageModel(nn.Module):
    # Apila embeddings de token + posición, varios bloques transformer,
    # una normalización final y una capa lineal que produce logits sobre el vocabulario.

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding por identidad del token
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # embedding por posición dentro del bloque
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # n_layer bloques transformer apilados
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # proyecta del espacio de embeddings al vocabulario (logits)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # inicializa pesos con una distribución normal de baja varianza; ayuda a que el entrenamiento arranque mejor
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # embedding según qué carácter es -> (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # embedding según la posición -> (T,C)
        x = tok_emb + pos_emb # se suman: cada token "sabe" quién es y dónde está -> (B,T,C)
        x = self.blocks(x) # pasa por todos los bloques transformer (atención + feedforward) -> (B,T,C)
        x = self.ln_f(x) # normalización final -> (B,T,C)
        logits = self.lm_head(x) # puntajes sin normalizar para cada carácter del vocabulario -> (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # aplanamos batch y tiempo para poder usar cross_entropy, que espera (N, C) y (N,)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # --

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # Genera texto carácter por carácter de forma autoregresiva: predice el siguiente,
        # lo agrega al contexto, y repite.
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # el modelo solo puede ver hasta block_size tokens de contexto
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # solo nos interesa la predicción del último paso de tiempo -> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # muestrea el siguiente carácter según las probabilidades -> (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # lo agrega a la secuencia para usarlo como contexto del siguiente paso -> (B, T+1)
        return idx


# ===================== Entrenamiento =====================
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # 10 million parameters

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # limpia los gradientes del paso anterior (set_to_none es más eficiente que ponerlos en cero)
    loss.backward() # backpropagation: calcula los gradientes
    optimizer.step() # actualiza los pesos del modelo

# ===================== Generación de texto con el modelo entrenado =====================
context = torch.zeros((1, 1), dtype=torch.long, device=device) # arranca la generación desde un solo carácter "vacío" (índice 0)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
