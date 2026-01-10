
def init(title, symbols, values):

    print(f"\ninitialized {title}:")
    print(f"{symbols[0]} = {values}")


def initzip(title, values, nb=None, prefix="", symbols=None):

    if nb is None:
        nb = len(values)

    if symbols is None:
        symbols = [f"{prefix}{i}" for i in range(1, nb+1)]

    print(f"\ninitialized {title}:")
    for symbol, value in zip(symbols, values):
        print(f"{symbol} = {value}")
