#ifndef BX_MATH_CU_H
#define BX_MATH_CU_H

__host__
double round_digits(double, int);

__host__
bool fuzzy_equals_digits(double, double, int);

__host__
void vec_relu_and_deriv(float*, float*, float*, int);

__host__
void vec_sigmoid_and_deriv(float*, float*, float*, int);

__host__
void normalize_ctf(float*, unsigned char*, int);

__host__
void vec_vec_outer(float*, float*, float*, int, int);

__host__
void mat_reduce_row_sum(float*, float*, int, int);

__host__
void mat_vec_multiply(float*, float*, int, int, float*);

__host__
void vec_vec_multiply(float*, float*, float*, int);

#endif
