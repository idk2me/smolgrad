from smolgrad.nn import MLP


def ascii_mlp(mlp: MLP) -> str:
    sizes = [len(mlp.layers[0].neurons[0].w)]
    sizes += [len(layer.neurons) for layer in mlp.layers]

    headers = [f"L{i}({s})" for i, s in enumerate(sizes)]
    # shared column width: widest of the neuron glyph and any header label
    col_w = max(len("( )"), max(len(h) for h in headers))
    sep = "  "

    max_n = max(sizes)
    cols: list[list[str]] = []
    for n in sizes:
        pad = (max_n - n) // 2
        cell = "( )".center(col_w)
        blank = " " * col_w
        col = [blank] * pad + [cell] * n + [blank] * (max_n - n - pad)
        cols.append(col)

    lines: list[str] = [sep.join(h.center(col_w) for h in headers)]
    for row in range(max_n):
        lines.append(sep.join(c[row] for c in cols))
    return "\n".join(lines)
