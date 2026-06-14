from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import override


class Op(StrEnum):
    ADD = "+"
    MUL = "*"
    POW = "^"
    ReLU = "ReLU"
    NO_OP = ""


class Value:
    def __init__(
        self,
        data: int | float,
        *,
        _prev: tuple[Value, ...] = (),
        _op: Op = Op.NO_OP,
        _label: str | None = None,
    ) -> None:
        self.data: int | float = data
        self.grad: float = 0.0
        self.prev: set[Value] = set(_prev)
        self._backward: Callable[[], None] = lambda: None
        self._op: Op = _op
        self._label: str | None = _label

    def __add__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            _prev=(self, other),
            _op=Op.ADD,
        )

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward

        return out

    def __mul__(self, other: int | float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            _prev=(self, other),
            _op=Op.MUL,
        )

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward

        return out

    def __pow__(self, other: int | float) -> Value:
        out = Value(
            self.data**other,
            _prev=(self,),
            _op=Op.POW,
        )

        def backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = backward

        return out

    def relu(self) -> Value:
        out = Value(
            (0 if self.data < 0 else self.data),
            _prev=(self,),
            _op=Op.ReLU,
        )

        def backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = backward

        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> Value:  # -self
        return self * -1

    def __radd__(self, other: int | float | Value) -> Value:
        return self + other

    def __sub__(self, other: int | float | Value) -> Value:
        return self + (-other)

    def __rsub__(self, other: int | float | Value) -> Value:
        return other + (-self)

    def __rmul__(self, other: int | float | Value) -> Value:
        return self * other

    def __truediv__(self, other: int | float | Value) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: int | float | Value) -> Value:
        return other * self**-1

    @override
    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
