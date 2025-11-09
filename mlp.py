from __future__ import annotations

import pathlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

NUM_HIDDEN_LAYERS = 5
MAX_NAME_LENGTH = 20
LOG_INTERVAL = 10000


class Dataset(NamedTuple):
    """NamedTuple representing a Dataset.

    Attributes:
        data: Training data.
        label: Corresponding label data.
    """

    data: jax.Array
    label: jax.Array


def compute_loss(model: MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute cross-entropy loss.

    Args:
        model: MLP model
        x: Input tensor of shape (batch_size, block_size)
        y: Target labels of shape (batch_size,)

    Returns:
        Scalar loss value
    """
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Perform single training step.

    Args:
        model: MLP model
        optimizer: Optimizer state
        x: Input batch of shape (batch_size, block_size)
        y: Target labels of shape (batch_size,)

    Returns:
        Loss value for this batch
    """
    loss, grads = nnx.value_and_grad(
        compute_loss, argnums=nnx.DiffState(0, nnx.Param)
    )(model, x, y)

    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: MLP, x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute loss on evaluation data.

    Args:
        model: MLP model
        x: Input batch of shape (batch_size, block_size)
        y: Target labels of shape (batch_size,)

    Returns:
        Loss value for this batch
    """
    return compute_loss(model, x, y)


def evaluate_dataset(
    model: MLP,
    dataset: Dataset,
    batch_size: int = 1024,
) -> float:
    """Evaluate model on a dataset using batched evaluation.

    Args:
        model: MLP model in eval mode
        dataset: Dataset to evaluate on
        batch_size: Batch size for evaluation

    Returns:
        Average loss over the dataset
    """
    num_samples = len(dataset.data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    total_loss = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        x_batch = dataset.data[start_idx:end_idx]
        y_batch = dataset.label[start_idx:end_idx]

        batch_loss = eval_step(model, x_batch, y_batch)
        total_loss += float(batch_loss) * (end_idx - start_idx)

    return total_loss / num_samples


def build_dataset(
    names: list[str], chars_to_idx: dict[str, int], block_size: int = 3
) -> Dataset:
    x = []
    y = []

    for name in names:
        context = [0] * block_size
        for char in name + ".":
            idx = chars_to_idx[char]
            x.append(context)
            y.append(idx)
            context = context[1:] + [idx]
    return Dataset(data=jnp.array(x), label=jnp.array(y))


def load_names(path: pathlib.Path) -> list[str]:
    if not path.exists():
        err = f"Could not find: {path}"
        raise FileNotFoundError(err)

    return path.read_text().splitlines()


def scale_last_layer_init(scale: float = 0.1):
    """Create a custom kernel initializer that scales down the initialization.

    This makes the model less confident initially, which can help with training stability.

    Args:
        scale: Factor to scale the initialization by

    Returns:
        Initializer function that can be passed to nnx.Linear
    """

    def init(key, shape, dtype=jnp.float32):
        return nnx.initializers.lecun_normal()(key, shape, dtype) * scale

    return init


class MLP(nnx.Module):
    """Multi-layer perceptron with embedding layer for character-level language modeling.

    Architecture:
    - Embedding layer: vocab_size -> n_embd
    - Multiple hidden layers: (Linear -> BatchNorm -> Tanh) x num_layers
    - Output layer: Linear -> BatchNorm
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int = 10,
        n_hidden: int = 100,
        num_layers: int = NUM_HIDDEN_LAYERS,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize MLP model.

        Args:
            vocab_size: Size of the vocabulary
            block_size: Number of characters in context window
            n_embd: Dimensionality of character embedding vectors
            n_hidden: Number of neurons in hidden layers
            num_layers: Number of hidden layers (default: 5)
            rngs: Random number generators for initialization
        """
        self.block_size = block_size
        self.n_embd = n_embd

        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=n_embd,
            rngs=rngs,
        )

        layers = []

        layers.extend(
            [
                nnx.Linear(
                    n_embd * block_size, n_hidden, use_bias=False, rngs=rngs
                ),
                nnx.BatchNorm(n_hidden, rngs=rngs),
                lambda x: jnp.tanh(x),
            ]
        )

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nnx.Linear(n_hidden, n_hidden, use_bias=False, rngs=rngs),
                    nnx.BatchNorm(n_hidden, rngs=rngs),
                    lambda x: jnp.tanh(x),
                ]
            )

        layers.extend(
            [
                nnx.Linear(
                    n_hidden,
                    vocab_size,
                    use_bias=False,
                    kernel_init=scale_last_layer_init(0.1),
                    rngs=rngs,
                ),
                nnx.BatchNorm(vocab_size, rngs=rngs),
            ]
        )

        self.mlp = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, block_size) containing character indices

        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        emb = self.embedding(x)
        x = emb.reshape(emb.shape[0], -1)

        return self.mlp(x)


def sample_name_mlp(
    model: MLP,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    rng_key: jax.Array,
    block_size: int,
) -> str:
    """Sample a name from the trained model.

    Args:
        model: Trained MLP model (should be in eval mode)
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        rng_key: JAX random key
        block_size: Size of context window

    Returns:
        Generated name string
    """
    name = []
    context = [char_to_idx["."]] * block_size

    while True:
        x = jnp.array([context])
        logits = model(x)[0]

        rng_key, subkey = jax.random.split(rng_key)
        idx = int(jax.random.categorical(subkey, logits))

        if idx == char_to_idx["."]:
            break

        name.append(idx_to_char[idx])
        context = context[1:] + [idx]

        if len(name) > MAX_NAME_LENGTH:
            break

    return "".join(name)


def main() -> None:
    block_size = 3
    n_embd = 10
    n_hidden = 100
    batch_size = 32
    max_steps = 200000
    learning_rate = 0.1
    lr_decay_step = 150000
    lr_decay_rate = 0.01
    # Set to None for full training
    debug_steps = 1000

    seed = 2147483647
    rng_key = jax.random.PRNGKey(seed)
    rng_key, model_key, shuffle_key, train_key = jax.random.split(rng_key, 4)

    names = load_names(pathlib.Path("./names.txt"))

    char_set = set()
    for name in names:
        char_set.update(name)
    chars = [".", *sorted(char_set)]
    char_to_idx = {s: i for i, s in enumerate(chars)}
    idx_to_char = {i: s for s, i in char_to_idx.items()}
    vocab_size = len(chars)

    num_names = len(names)
    indices = jax.random.permutation(shuffle_key, num_names)
    names = [names[int(i)] for i in indices]

    names_80 = int(0.8 * num_names)
    names_90 = int(0.9 * num_names)

    train_dataset = build_dataset(
        names=names[:names_80], chars_to_idx=char_to_idx, block_size=block_size
    )
    val_dataset = build_dataset(
        names=names[names_80:names_90],
        chars_to_idx=char_to_idx,
        block_size=block_size,
    )
    test_dataset = build_dataset(
        names=names[names_90:], chars_to_idx=char_to_idx, block_size=block_size
    )

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(train_dataset.data)}")
    print(f"Validation examples: {len(val_dataset.data)}")
    print(f"Test examples: {len(test_dataset.data)}")

    model = MLP(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_hidden=n_hidden,
        rngs=nnx.Rngs(model_key),
    )

    total_params = sum(
        x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
    print(f"Total parameters: {total_params}")

    schedule = optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales={lr_decay_step: lr_decay_rate / learning_rate},
    )

    optimizer = nnx.Optimizer(model, optax.sgd(schedule), wrt=nnx.Param)

    actual_steps = debug_steps if debug_steps is not None else max_steps
    print(f"\nTraining for {actual_steps} steps...")

    for step in range(actual_steps):
        train_key, subkey = jax.random.split(train_key)
        indices = jax.random.randint(
            subkey,
            shape=(batch_size,),
            minval=0,
            maxval=train_dataset.data.shape[0],
        )
        x_batch = train_dataset.data[indices]
        y_batch = train_dataset.label[indices]

        loss = train_step(model, optimizer, x_batch, y_batch)

        if step % LOG_INTERVAL == 0:
            print(f"{step:7d}/{actual_steps:7d}: {loss:.4f}")

    print("\nFinal losses:")

    model.eval()
    train_loss = evaluate_dataset(model, train_dataset, batch_size=1024)
    val_loss = evaluate_dataset(model, val_dataset, batch_size=1024)
    test_loss = evaluate_dataset(model, test_dataset, batch_size=1024)
    print(f"Train: {train_loss:.4f}")
    print(f"Val: {val_loss:.4f}")
    print(f"Test: {test_loss:.4f}")

    print("\nSample generated names:")
    rng_key, sample_key = jax.random.split(rng_key)
    for _ in range(10):
        sample_key, subkey = jax.random.split(sample_key)
        name = sample_name_mlp(
            model, char_to_idx, idx_to_char, subkey, block_size
        )
        print(f"  {name}")


if __name__ == "__main__":
    main()
