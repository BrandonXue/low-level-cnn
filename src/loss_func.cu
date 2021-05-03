#include <math.h>

#include "loss_func.cu.h"

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

// I am using the exact fuzz factor used by Keras and TF
static const double EPSILON_FUZZ = 0.0000001;

/**
 * Categorical Cross Entropy Loss.
 */
__host__
float cat_cross_entropy(float *y_true, float *y_pred, int len) {
    // To calculate - summation p(y_true) * ln(y_pred)
    // The predictions have to be adjusted so that they add to 1,
    // so that they are a valid probability distribution.
    float sum = 0;
    for (int i = 0; i < len; ++i) {
        sum += y_pred[i];
    }
    float y_pred_norm[len];
    for (int i = 0; i < len; ++i) {
        y_pred_norm[i] = y_pred[i] / sum;
    }
    float result = 0;
    for (int i = 0; i < len; ++i) {
        // Natural log of 0 is negative inf, so prevent that
        result += y_true[i] *
            log (
                (y_pred_norm[i] < EPSILON_FUZZ) ?
                EPSILON_FUZZ : y_pred_norm[i]
            );
             
    }
    return result;
}
