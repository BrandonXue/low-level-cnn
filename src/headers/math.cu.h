#ifndef BX_MATH_H
#define BX_MATH_H

/* ============================= Math Utilities =============================*/

__host__
int argmax(float*, int);

__host__
bool fuzzy_equals_digits(double, double, int);

__host__
float rand_float(float, float);

__host__
double round_digits(double, int);

/* ======================== Vector/Matrix Operations ========================*/

__host__
void print_vec(float*, int, int);

__host__
void random_init(float*, int, float, float);

__host__
void vec_relu_and_deriv(float*, float*, float*, int);

__host__
void vec_sigmoid_and_deriv(float*, float*, float*, int);

__host__
void normalize_ctf(float*, unsigned char*, int);

__host__
void vec_vec_outer(float*, float*, float*, int, int);

__host__
void mat_vec_dot(float *, float *, int, int, float*);

__host__
void vec_vec_multiply(float*, float*, float*, int);

__host__
void vec_mat_dot(float*, float*, float*, int, int);

__host__
float convolution_2d(
    int, int,
    float*, int, int,
    float*, int, int,
    int, int
);

__host__
void all_convolution_2d(
    float*,
    float*, int, int,
    float*, int, int, int,
    int, int
);

__host__
float back_convolution_2d(
    int, int,
    float*, int, int,
    float*, int, int,
    int, int
);

void all_back_convolution_2d(
    float*, int, int,
    float*, int, int,
    float*, int, int, int,
    int, int
);

/* ============================= Loss functions =============================*/

__host__
float mse(float*, float*, int);

__host__
void cat_cross_entropy(int, float*, float*, float*, float*, float*);

#endif
