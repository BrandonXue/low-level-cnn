#include <float.h>
#include <math.h>

#include <stdio.h>

#include "math.cu.h"



__host__
double round_digits(double x, int digits) {
    double mul_factor = pow(10, digits);
    return round(x * mul_factor) / mul_factor;
}

__host__
bool fuzzy_equals_digits(double a, double b, int digits) {
    return round_digits(a, digits) == round_digits(b, digits);
}

/**
 * Element-wise multiplication of two vectors.
 * Output does not need zero-initialization.
 */
__host__
void vec_vec_multiply(float *out, float *a, float*b, int len) {
    for (int i = 0; i < len; ++i) {
        out[i] = a[i] * b[i];
    }
}

/**
 * Dot product of an M x N  matrix A, and a vector v's diagonal.
 * out = A • diag(v).
 * Basically multiplies each column of A by one element of v.
 * out should have rank of 1 (an array interpreted as a matrix).
 * out does not need zero-initialization.
 */
__host__
void mat_vec_multiply(float *out, float *A, int M, int N, float *v) {
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < M; ++row) {
            out[row * N + col] = A[row * N + col] * v[col];
        }
    }
}

/**
 * Reduce an M x N  matrix into a vector by the addition operation.
 * Reduces by adding elements in the same row together.
 * Output does not need zero-initialization.
 */
__host__
void mat_reduce_row_sum(float *out, float *A, int M, int N) {
    for (int row = 0; row < M; ++row) {
        out[row] = 0;
        for (int col = 0; col < N; ++col) {
            out[row] += A[row * N + col];
        }
    }
}

/**
 * Outer product of two vectors. The output has rows of b, each multiplied by
 * a factor of an element of a.    a = [a1  a2  a3 ... aM]    b = [b1  b2  b3 ... bN]
 * out = [a1b1 a1b2 a1b3 ... a1bN
 *        a2b1 a2b2 a2b3 ... a2bN
           .    .    .    .    .
          aMb1 aMb2 aMb3 ... aMbN]
 */
__host__
void vec_vec_outer(float *out, float *a, float *b, int a_len, int b_len) {
    for (int row = 0; row < a_len; ++row) {
        for (int col = 0; col < b_len; ++col) {
            out[row * b_len + col] = a[row] * b[col];
        }
    }
}

/**
 * Normalize an image with unsigned char (1 byte) to floating point (4 bytes),
 * with min 0 and range 1.
 */
__host__
void normalize_ctf(float *res, unsigned char *arr, int len) {
    int max = 0;
    int min = 255;
    for (int i = 0; i < len; ++i) {
        if ((int)arr[i] > max) {
            max = (int)arr[i];
        }
        if ((int)arr[i] < min) {
            min = (int)arr[i];
        }
    }
    if (max <= min) {
        printf("Warning: normalize had invalid denominator.\n");
    }
    for (int i = 0; i < len; ++i) {
        res[i] = ((float)arr[i] - min) / (float)(max - min);
    }
}

/**
 * Vectorized sigmoid and its first derivative.
 */
__host__
void vec_sigmoid_and_deriv(float *out, float *out_deriv, float *in, int len) {
    for (int i = 0; i < len; ++i) {
        if (in[i] < 0) {
            float exp_x = exp(in[i]);
            out[i] = exp_x / (1 + exp_x);
        } else {
            out[i] = 1 / (1 + exp(-in[i]));
        }
        //out[i] = 1 / (1 + exp(-in[i]));
        out_deriv[i] = out[i] * (1 - out[i]);
    }
}

/**
 * Vectorized ReLU and its first derivative.
 */
__host__
void vec_relu_and_deriv(float *out, float *out_deriv, float *in, int len) {
    for (int i = 0; i < len; ++i) {
        if (in[i] > 0) {
            out[i] = in[i];
            out_deriv[i] = 1;
        } else {
            out[i] = 0;
            out_deriv[i] = 0;
        }
    }
}


