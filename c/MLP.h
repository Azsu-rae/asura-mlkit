#ifndef MLP_H
#define MLP_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>

#include "./dsa/arrays.h"

#define MLP_CTOR(...) MLP_ctor(__VA_ARGS__, -1)

typedef struct {
    size_t* shape;
    double*** weights;
    double** bias;
} MLP;


#endif
