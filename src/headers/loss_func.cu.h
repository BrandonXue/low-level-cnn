#ifndef LOSS_FUNC_H
#define LOSS_FUNC_H

__host__
float mse(float*, float*, int);

__host__
float cat_cross_entropy(float*, float*, int);

#endif
