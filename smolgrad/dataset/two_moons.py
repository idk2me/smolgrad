import random
import math

from smolgrad import Value


def make_moons(n: int, noise: float = 0.1) -> tuple[list[list[Value]], list[int]]:
    xs: list[list[Value]] = []
    ys: list[int] = []
    for i in range(n):
        if i < n // 2:
            angle = math.pi * (i / (n // 2))
            x, y = math.cos(angle), math.sin(angle)
            label = 0
        else:
            j = i - n // 2
            angle = math.pi * (j / (n - n // 2))
            x, y = 1 - math.cos(angle), 0.5 - math.sin(angle)
            label = 1
        x += random.gauss(0, noise)
        y += random.gauss(0, noise)
        xs.append([Value(x), Value(y)])
        ys.append(label)
    return xs, ys


def train_test_split(
    xs: list[list[Value]],
    ys: list[int],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[list[Value]], list[int], list[list[Value]], list[int]]:
    rng = random.Random(seed)
    idx = list(range(len(xs)))
    rng.shuffle(idx)
    cut = int(len(xs) * (1 - test_frac))
    train_i, test_i = idx[:cut], idx[cut:]
    return (
        [xs[i] for i in train_i],
        [ys[i] for i in train_i],
        [xs[i] for i in test_i],
        [ys[i] for i in test_i],
    )


def get_training_split(
    n: int = 200,
    noise: float = 0.1,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[list[Value]], list[int], list[list[Value]], list[int]]:
    xs, ys = make_moons(n, noise)
    return train_test_split(xs, ys, test_frac, seed)
