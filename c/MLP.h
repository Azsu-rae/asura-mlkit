#ifndef MLP_H
#define MLP_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#include "./dsa/Data.h"
#include "Tensor.h"

#define MLP_CTOR(...) MLP_ctor(DOUBLE, __VA_ARGS__, -1)

typedef struct {} MLP;

MLP MLP_ctor(DType type, size_t layer, ...);

#endif
