#ifndef BX_NN_LAYERS_H
#define BX_NN_LAYERS_H

__host__
int calc_dims_pad_valid(int, int, int);

__host__
void Conv2D_forward(
    float*, int, int,
    float*,
    float*,
    float*, int, int,
    float*, int, int,
    int, int, int,
    int
);

__host__
void Conv2D_backward(
    int, int,
    float*,
    float*,
    float*,
    float*,
    float*, int, int,
    float*, int, int,
    float*,
    int, int, int
);

__host__
void Dense_forward(
    float*, int,
    float*,
    float*,
    float*, int,
    float*,
    int
);

__host__
void Dense_backward(
    int,
    float*,
    float*,
    float*,
    float*,
    float*, int,
    float*,
    float*,
    int
);

__host__
void SGD_update_params(float, float*, float*, int);

#endif
