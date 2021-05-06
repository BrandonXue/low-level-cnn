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
