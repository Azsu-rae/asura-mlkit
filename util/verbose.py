
def write(s, verb):
    if verb is True:
        print(s)


def init(title, symbols, values, verb=True):

    write(f"\ninitialized {title}:", verb)
    write(f"{symbols[0]} = {values}", verb)


def initzip(title, values, nb=None, prefix="", symbols=None, verb=True):

    if nb is None:
        nb = len(values)

    if symbols is None:
        symbols = [f"{prefix}{i}" for i in range(1, nb+1)]

    write(f"\ninitialized {title}:\n", verb)
    for symbol, value in zip(symbols, values):
        write(f"{symbol} = {value}", verb)


def start(s, verb=True):
    write(f"\n{s}", verb)


def step(name, i, verb=True):
    write(f"\n{name} {i}:", verb)


def writeIf(cond, true, false, verb=True):
    if cond:
        write(f"\n{true}", verb)
    else:
        write(f"\n{false}", verb)



def equation(formula, substituted, result, cond=True, verb=True):
    if cond:
        write(f"\n{formula}", verb)
        write(f"{substituted} = {result}", verb)


def state(symbol, values, verb=True):
    write(f"{symbol} = {values}", verb)


def pad(verb=True):
    write("", verb)
