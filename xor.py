from smolgrad import Value, ascii_mlp
from smolgrad.nn import MLP

xs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

ys = [0, 1, 1, 0]

model = MLP(2, [5, 1])  # works better than [4,1]
print(ascii_mlp(model))

lr = 0.05

for step in range(200):
    preds = [model([Value(x[0]), Value(x[1])])[0] for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    if not isinstance(loss, Value):
        loss = Value(loss)

    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"Step {step} completed...")

final_preds = [model([Value(x[0]), Value(x[1])])[0] for x in xs]

for idx, val in enumerate(final_preds):
    print(
        f"{xs[idx]} -> {val.data:.4f} ->",
        f"{1 if val.data >= 0.5 else 0}: Expected {ys[idx]}",
    )
