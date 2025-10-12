import itertools
import pathlib

import jax.numpy as jnp
import jax

Bigrams = list[tuple[int, int]]


def loss_fn(weights, x, y, num_classes: int):
    x_encoded = jax.nn.one_hot(x, num_classes=num_classes)
    logits = x_encoded @ weights
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(len(x)), y].mean()


def sample_name(weights, char_to_idx, idx_to_char, rand_key, num_classes: int):
    name = []
    idx = char_to_idx["."]

    while True:
        x_encoded = jax.nn.one_hot(idx, num_classes=num_classes)
        logits = x_encoded @ weights
        log_probs = jax.nn.log_softmax(logits)

        rand_key, subkey = jax.random.split(rand_key)
        idx = jax.random.categorical(subkey, log_probs)

        if idx == char_to_idx["."]:
            break

        name.append(idx_to_char[int(idx)])

    return "".join(name), rand_key


def load_names(path: pathlib.Path) -> list[str]:
    if not path.exists():
        err = f"Could not find: {path}"
        raise FileNotFoundError(err)

    return path.read_text().splitlines()


def get_train_test_data(
    bigrams: Bigrams,
) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    x_, y_ = zip(*bigrams, strict=True) if bigrams else ([], [])
    return jnp.array(x_), jnp.array(y_)


def main():
    names = load_names(pathlib.Path("./names.txt"))

    chars = [".", *sorted(set("".join(names)))]
    char_to_idx = {s: i for i, s in enumerate(chars)}
    idx_to_char = {i: s for s, i in char_to_idx.items()}
    vocab_size = len(chars)

    bigrams = []
    for name in names:
        name_chars = [".", *list(name), "."]
        bigrams.extend(
            (char_to_idx[x_char], char_to_idx[y_char])
            for x_char, y_char in itertools.pairwise(name_chars)
        )

    x, y = get_train_test_data(bigrams)
    rand_key = jax.random.key(123)
    rand_key, subkey = jax.random.split(rand_key)
    weights = jax.random.normal(key=subkey, shape=(vocab_size, vocab_size))

    learning_rate = 50.0

    for iteration in range(1000):
        loss, grads = jax.value_and_grad(loss_fn)(
            weights, x, y, num_classes=vocab_size
        )

        weights = weights - learning_rate * grads

        if iteration % 10 == 0:
            print(f"iteration {iteration}: loss = {loss.item():.4f}")

    print("\nGenerated names:")
    rand_key = jax.random.key(456)
    for i in range(5):
        name, rand_key = sample_name(
            weights, char_to_idx, idx_to_char, rand_key, vocab_size
        )
        print(f"{i + 1}. {name.capitalize()}")


if __name__ == "__main__":
    main()
