"""Unit tests for smolgrad.nn (Module / Neuron / Layer / MLP)."""

import random
import unittest

from smolgrad import Value
from smolgrad.nn import MLP


def mlp_loss(
    mlp: MLP,
    x_vals: list[float],
    w_vals: list[float],
    targets: list[float],
) -> tuple[Value, list[Value]]:
    """Run one forward pass with weights written into the parameters.

    Returns the summed squared error and the input Values, so callers can
    inspect gradients of both parameters and inputs.
    """
    for p, w in zip(mlp.parameters(), w_vals):
        p.data = w
    x = [Value(v) for v in x_vals]
    out = mlp(x)
    total = Value(0.0)
    for o, t in zip(out, targets):
        total = total + (o - t) ** 2
    return total, x


class TestMLP(unittest.TestCase):
    def test_forward_shape(self) -> None:
        mlp = MLP(2, [3, 1])
        out = mlp([Value(1.0), Value(-2.0)])
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], Value)
        self.assertIsInstance(out[0].data, (int, float))

    def test_parameters_count(self) -> None:
        # 4 neurons x (2 weights + 1 bias) + 3 x (4 + 1) + 1 x (3 + 1) = 31
        mlp = MLP(2, [4, 3, 1])
        self.assertEqual(len(mlp.parameters()), 31)

    def test_zero_grad(self) -> None:
        mlp = MLP(2, [3, 1])
        out = mlp([Value(1.0), Value(-2.0)])
        out[0].backward()
        self.assertTrue(any(p.grad != 0 for p in mlp.parameters()))
        mlp.zero_grad()
        self.assertTrue(all(p.grad == 0 for p in mlp.parameters()))

    def test_gradients_match_finite_differences(self) -> None:
        # The strongest check: every analytic gradient produced by
        # backward() must match a central finite difference of the
        # forward pass, for all parameters and all inputs.
        rng = random.Random(42)
        mlp = MLP(3, [4, 2])
        targets = [0.5, -1.0]
        w_vals = [rng.uniform(-1.5, 1.5) for _ in mlp.parameters()]
        x_vals = [rng.uniform(-2.0, 2.0) for _ in range(3)]
        h = 1e-6

        def forward(inputs: list[float], weights: list[float]) -> float:
            return mlp_loss(mlp, inputs, weights, targets)[0].data

        loss, x = mlp_loss(mlp, x_vals, w_vals, targets)
        loss.backward()
        param_grads = [p.grad for p in mlp.parameters()]
        input_grads = [v.grad for v in x]

        for i in range(len(w_vals)):
            w_plus, w_minus = w_vals.copy(), w_vals.copy()
            w_plus[i] += h
            w_minus[i] -= h
            numeric = (
                forward(x_vals, w_plus) - forward(x_vals, w_minus)
            ) / (2 * h)
            self.assertAlmostEqual(param_grads[i], numeric, places=4)

        for i in range(len(x_vals)):
            x_plus, x_minus = x_vals.copy(), x_vals.copy()
            x_plus[i] += h
            x_minus[i] -= h
            numeric = (
                forward(x_plus, w_vals) - forward(x_minus, w_vals)
            ) / (2 * h)
            self.assertAlmostEqual(input_grads[i], numeric, places=4)

    def test_training_reduces_loss(self) -> None:
        # Fit y = 2x + 1 with a single linear output neuron. The problem
        # is convex, so plain gradient descent must reduce the loss.
        mlp = MLP(1, [1])
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [3.0, 5.0, 7.0, 9.0]

        def mse() -> Value:
            total = Value(0.0)
            for xv, yv in zip(xs, ys):
                pred = mlp([Value(xv)])[0]
                total = total + (pred - yv) ** 2
            return total

        before = mse().data
        for _ in range(100):
            loss = mse()
            mlp.zero_grad()
            loss.backward()
            for p in mlp.parameters():
                p.data -= 0.01 * p.grad
        after = mse().data

        self.assertLess(after, before)
