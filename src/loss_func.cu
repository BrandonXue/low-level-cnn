#include <math.h>

#include "loss_func.cu.h"

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
