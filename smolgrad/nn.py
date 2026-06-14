from typing import override
from smolgrad import Value
import random


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, n_in: int, non_linear: bool = True) -> None:

        self.w: list[Value] = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b: Value = Value(0)
        self.non_linear: bool = non_linear

    def __call__(self, x: list[Value]) -> Value:
        act: Value = self.b
        for wi, wx in zip(self.w, x):
            act += wi * wx
        return act.relu() if self.non_linear else act

    @override
    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    @override
    def __repr__(self) -> str:
        return f"{'ReLU' if self.non_linear else 'Linear'} Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, non_linear: bool) -> None:
        self.neurons: list[Neuron] = [Neuron(n_in, non_linear) for _ in range(n_out)]

    def __call__(self, x: list[Value]) -> list[Value]:
        out: list[Value] = [n(x) for n in self.neurons]
        return out

    @override
    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    @override
    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, n_in: int, n_out: list[int]) -> None:
        sz = [n_in] + n_out
        self.layers: list[Layer] = [
            Layer(sz[i], sz[i + 1], (i != len(n_out) - 1)) for i in range(len(n_out))
        ]

    def __call__(self, x: list[Value]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    @override
    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    @override
    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
