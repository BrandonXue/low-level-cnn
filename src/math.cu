#include <float.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

#include "math.cu.h"

/**
 * Pretty print a vector. Useful for debug.
 */
__host__
void print_vec(float *vec, int len, int precision) {
    if (precision < 0) {
        precision = 0;
    }

    printf("[");
    for (int i = 0; i < len; ++i) {
        printf("%0.*f", precision, vec[i]);
        if (i + 1 < len) {
            printf("  ");
        }
    }
    printf("]\n");
}

__host__
float rand_float(float min, float max) {
    double range = max - min;
    double zero_to_one = (double)rand() / (double)(RAND_MAX);
    return float(zero_to_one * range + (double)min);
}

__host__
void random_init(float *arr, int len, float min, float max) {
    for (int i = 0; i < len; ++i ) {
        arr[i] = rand_float(min, max);
    }
}

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
 * A combined operation on matrix A and vector v,
 * A • diag(v) reduced by sum (adding each row's elements together).
 */
__host__
void mat_vec_multiply_reduce_sum(float *out, float *A, int M, int N, float *v) {
    for (int row = 0; row < M; ++row) {
        out[row] = 0;
        for (int col = 0; col < N; ++col) {
            out[row] += A[row * N + col] * v[col];
        }
    }
}

/**
 * Dot product between row vector v and matrix A.
 * Output does not need zero-initialization.
 */
__host__
void vec_mat_dot(float *out, float *v, float *A, int M, int N) {
    for (int mat_col = 0; mat_col < N; ++mat_col) {
        out[mat_col] = 0;
        for (int mat_row = 0; mat_row < M; ++mat_row) {
            out[mat_col] += v[mat_row] * A[mat_row * N + mat_col];
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

// I am using the exact fuzz factor used by Keras and TF
static const float EPSILON_FUZZ = 0.0000001;

/**
 * Calculate the mean squared error.
 * @param y_true Pointer to an array of actual y values.
 * @param y_pred Pointer to an array of observed y values.
 * @len The length of both y_true and y_pred.
 */
__host__
float  mse(float *y_true, float *y_pred, int len) {
    float sse = 0;
    for (int i = 0; i < len; ++i) {
        // Error
        float temp  = y_true[i] - y_pred[i];
        // Squared error
        temp *= temp;
        // Sum up the squared error (TODO: overflow underflow protection, FMA)
        sse += temp;
    }
    // Mean squared error
    return sse / len;
}


/**
 * Categorical Cross Entropy Loss. Assumes that the inputs
 * have not gone through any activation function yet, and applies
 * the softmax activation as part of this function call.
 * Also calculates the change in loss wrt the values before activation.
 */
__host__
void cat_cross_entropy(
    int len, float *y_true, float *vals, float *outs, float *pdL_pdval, float *loss
) {
    // softmax(x) = softmax(x - C)
    // For numeric stability, we will subtract the max from each element.
    // First find the max of the un-activated values.
    float vals_max = vals[0];
    for (int i = 0; i < len; ++i) {
        if (vals[i] > vals_max) {
            vals_max = vals[i];
        }
        //printf("in softmax val[%d] = %f\n", i, vals[i]);
    }

    // Find the numerator and the denominator
    // Numerator is e^(x - max_x) denominator is the summation of all the numerators
    float sum_exp = 0;
    for (int i = 0; i < len; ++i) {
        outs[i] = exp(vals[i] - vals_max);
        sum_exp += outs[i];
    }
    // In the unlikely event the denom is zero, add fuzz factor
    if (sum_exp == 0) {
        sum_exp = EPSILON_FUZZ;
    }
    // Finish calculating the softmaxes by dividing each element by the denominator
    for (int i = 0; i < len; ++i) {
        outs[i] = outs[i] / sum_exp;
    }

    // Calculate the Cross Entropy Loss, -Σ (y_true * log(softmax(vals))
    *loss = 0;
    for (int i = 0; i < len; ++i) {
        *loss += y_true[i] * log(outs[i]);
    }
    // There is a negative one coefficient.
    *loss *= -1;

    // Calculate the gradient
    // For each element, the derivative is softmax(x)_i  -  y_true_i
    for (int i = 0; i < len; ++i) {
        pdL_pdval[i] = outs[i] - y_true[i];
    }
}

