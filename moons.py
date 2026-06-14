import gc
import sys

from smolgrad.dataset import two_moons
from smolgrad.nn import MLP
from smolgrad import Value, ascii_mlp

model = MLP(2, [8, 16, 16, 1])
print(ascii_mlp(model))

lr = 0.05
epochs = 200


def main(no_gc: bool = False):
    # runs quicker
    if no_gc:
        gc.disable()

    X_train, y_train, X_test, y_test = two_moons.get_training_split()

    for e in range(epochs):
        preds = [model([x[0], x[1]])[0] for x in X_train]
        loss = sum((p - y) ** 2 for p, y in zip(preds, y_train)) / len(y_train)

        if not isinstance(loss, Value):
            loss = Value(loss)

        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

        print(f"Epoch {e}: loss = {loss.data:.4f}")

    print(f"Completed training with {epochs} epochs...")

    final_preds = [model([x[0], x[1]])[0] for x in X_test]

    correct = 0
    for idx, val in enumerate(final_preds):
        x0, x1 = X_test[idx][0].data, X_test[idx][1].data
        expected = y_test[idx]
        predicted = 1 if val.data > 0.5 else 0
        is_right = predicted == expected
        correct += is_right
        print(
            f"({x0:+.2f}, {x1:+.2f}) -> raw={val.data:+.3f} -> ",
            f"pred={predicted} expected={expected} {'O' if is_right else 'X'}",
        )

    accuracy = correct / len(y_test)
    print(f"\nTest accuracy: {correct}/{len(y_test)} = {accuracy:.1%}")


if __name__ == "__main__":
    args = sys.argv
    main(no_gc="--no-gc" in args)
