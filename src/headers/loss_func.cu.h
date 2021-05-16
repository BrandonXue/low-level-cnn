#ifndef LOSS_FUNC_H
#define LOSS_FUNC_H

__host__
float mse(float*, float*, int);

__host__
void cat_cross_entropy(int, float*, float*, float*, float*, float*);

__host__
void cn_mnist_target_to_vec(float*, int);

#endif
