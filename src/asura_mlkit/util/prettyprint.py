

import torch


def tensor_add(t1: torch.Tensor, t2: torch.Tensor) -> str:
    if t1.shape != t2.shape:
        raise ValueError("BOO!")
    elif len(t1.shape) == 0:
        return f"{float(t1)} + {float(t2)}"

    size = t1.shape[0]
    els = []
    for fiber in range(size):
        s = travel_tensors(t1[fiber], t2[fiber])
        tmp = s.split("],")
        if tmp:
            s = "],\n".join(tmp)
        els.append(s)

    return "[" + ", ".join(els) + "]"
