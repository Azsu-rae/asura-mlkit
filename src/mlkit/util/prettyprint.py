

import torch

import re

def _tensor_add(t1: torch.Tensor, t2: torch.Tensor) -> str:
    if t1.shape != t2.shape:
        raise ValueError("BOO!")
    elif len(t1.shape) == 0:
        return f"{float(t1):.4f} + {float(t2):.4f}"

    size = t1.shape[0]
    els: list[str] = []
    for fiber in range(size):
        els.append(_tensor_add(t1[fiber], t2[fiber]))

    result = "[" + ", ".join(els) + "]"

    result = re.sub(r"((?<=\n)\s*\[\n)", r"  \1", result)
    result = re.sub(r"((?<=\n)\s*\]\n)", r"  \1", result)

    # special case
    result = re.sub(r"(?<=(?:(?<!,) \[\n)|\[\[\n)(\s*\[\d)", r"  \1", result)

    result = re.sub(r"\[\[\n", r"[\n  [\n", result)
    result = re.sub(r"\n]]", r"\n  ]\n]", result)

    result = re.sub(r"],\n", r"],\n  ", result)
    result = re.sub(r"(?<=\n)(\s*], \[\n)", r"  \1  ", result)

    # BASE CASE
    result = re.sub(r"\[\[(?=\d)", r"[\n  [", result)
    result = re.sub(r"(?<=\d)]]", "]\n]", result)
    result = re.sub(r"(?<=\d)], ", "],\n  ", result)

    return result

def tensor_add(t1: torch.Tensor, t2: torch.Tensor):
    print(_tensor_add(t1, t2))
