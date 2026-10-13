"""Unit tests for the scalar autograd engine (smolgrad.engine.Value)."""

import unittest

from smolgrad import Value


class TestForwardOps(unittest.TestCase):
    """Forward passes for every operator, including reflected variants."""

    def test_arithmetic(self) -> None:
        self.assertEqual((Value(2) + Value(3)).data, 5)
        self.assertEqual((Value(2) + 3).data, 5)
        self.assertEqual((2 + Value(3)).data, 5)  # __radd__
        self.assertEqual((Value(2) * Value(-4)).data, -8)
        self.assertEqual((3 * Value(2)).data, 6)  # __rmul__
        self.assertEqual((Value(2) - Value(5)).data, -3)
        self.assertEqual((5 - Value(2)).data, 3)  # __rsub__
        self.assertEqual((-Value(2)).data, -2)  # __neg__
        self.assertAlmostEqual((Value(7) / Value(2)).data, 3.5)
        self.assertAlmostEqual((7 / Value(2)).data, 3.5)  # __rtruediv__
        self.assertEqual((Value(2) ** 3).data, 8)

    def test_relu(self) -> None:
        self.assertEqual(Value(-3).relu().data, 0)
        self.assertEqual(Value(0).relu().data, 0)
        self.assertEqual(Value(2.5).relu().data, 2.5)


class TestBackward(unittest.TestCase):
    """Hand-computed gradients for the core operators."""

    def test_simple_ops(self) -> None:
        # f = a * b + c: df/da = b, df/db = a, df/dc = 1
        a, b, c = Value(2), Value(-3), Value(10)
        (a * b + c).backward()
        self.assertEqual(a.grad, -3)
        self.assertEqual(b.grad, 2)
        self.assertEqual(c.grad, 1)

        # f = x**2 at x = 3: df/dx = 2x = 6
        x = Value(3)
        (x**2).backward()
        self.assertEqual(x.grad, 6)

        # f = a / b at (a, b) = (6, 3):
        # df/da = 1/b = 1/3, df/db = -a/b**2 = -2/3
        a, b = Value(6), Value(3)
        (a / b).backward()
        self.assertAlmostEqual(a.grad, 1 / 3)
        self.assertAlmostEqual(b.grad, -2 / 3)

    def test_repeated_use_accumulates(self) -> None:
        # f = a*a + b*b at (a, b) = (3, 4): df/da = 2a = 6, df/db = 2b = 8.
        # a and b each appear twice in the graph, so their grads must
        # accumulate through both paths.
        a, b = Value(3), Value(4)
        f = a * a + b * b
        f.backward()
        self.assertEqual(f.data, 25)
        self.assertEqual(a.grad, 6)
        self.assertEqual(b.grad, 8)

    def test_relu_grads(self) -> None:
        # relu'(x) = 1 for x > 0, else 0 (subgradient 0 at exactly 0)
        for x_val, expected in [(2.0, 1.0), (-2.0, 0.0), (0.0, 0.0)]:
            x = Value(x_val)
            y = x.relu()
            y.backward()
            self.assertEqual(x.grad, expected)

    def test_composite_expression(self) -> None:
        # f = d * e + a * a, where d = a * b + c and e = a * c + b.
        # Every leaf fans out into multiple subexpressions.
        a, b, c = Value(2.0), Value(-3.0), Value(4.0)
        d = a * b + c
        e = a * c + b
        f = d * e + a * a

        self.assertEqual(f.data, -6.0)

        f.backward()

        # df/dd = e = 5, df/de = d = -2
        self.assertEqual(d.grad, 5)
        self.assertEqual(e.grad, -2)
        # a fans out into d, e and a*a: 5*(-3) + (-2)*4 + 2*2 = -19
        self.assertEqual(a.grad, -19)
        # b fans out into d and e: 5*2 + (-2)*1 = 8
        self.assertEqual(b.grad, 8)
        # c fans out into d and e: 5*1 + (-2)*2 = 1
        self.assertEqual(c.grad, 1)
