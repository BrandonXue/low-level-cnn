#include "catch.hpp"

#include "math.cu.h"

TEST_CASE( "random floats are bounded properly", "[math]" ) {
    for (int i = 0; i < 100; ++i) {
        float my_rand = rand_float(-0.1, 0.1);
        REQUIRE( my_rand >= -0.1 );
        REQUIRE( my_rand <= 0.1 );
    }

    float my_weights[10];
    random_init(my_weights, 10, -0.5, 0.5);
    for (int i = 0; i < 10; ++i) {
        REQUIRE( my_weights[i] >= -0.5 );
        REQUIRE( my_weights[i] <= 0.5 );
    }
}

TEST_CASE( "round by digits works", "[math]" ) {
    REQUIRE( round_digits(4.2345678, 3) == 4.235 );
    REQUIRE( round_digits(4.2345678, 9) == 4.2345678 );
    REQUIRE( round_digits(4.2345678, 0) == 4.0 );
}

TEST_CASE( "round by digits precision not too awful", "[math]" ) {
    REQUIRE( 
        round_digits( 900000000000000000000000.1234567, 5) 
        == 900000000000000000000000.0
    );
}

TEST_CASE( "fuzzy equals accepts within tolerance", "[math]") {
    REQUIRE( fuzzy_equals_digits(3.44444, 3.44445, 3) );
    REQUIRE( fuzzy_equals_digits(12.3, 12.4, 0) );
    REQUIRE( fuzzy_equals_digits(0.0000000001, 0.00000000005, 10) );
}

TEST_CASE( "fuzzy equals rejects exceeding tolerance", "[math]" ) {
    REQUIRE( !fuzzy_equals_digits(3.44444, 3.44445, 4) );
    REQUIRE( !fuzzy_equals_digits(12.3, 12.4, 2) );
    REQUIRE( !fuzzy_equals_digits(0.0000000001, 0.00000000004, 10) );
}

TEST_CASE( "vector and vector element-wise multiplication", "[math]" ) {
    float a[5] = {2.1, 1.8, 1.2, 0.4, 0.8};
    float b[5] = {1.0, 1.1, 2.4, 3.1, 1.9};
    float c[5] = {0.3, 0.1, 4.1, 2.3, 1.4};
    float resbuf_ab[5];
    float resbuf_bc[5];
    float resbuf_ac[5];
    float expect_ab[5] = {2.1, 1.98, 2.88, 1.24, 1.52};
    float expect_bc[5] = {0.3, 0.11, 9.84, 7.13, 2.66};
    float expect_ac[5] = {0.63, 0.18, 4.92, 0.92, 1.12};
    
    vec_vec_multiply(resbuf_ab, a, b, 5);
    vec_vec_multiply(resbuf_bc, b, c, 5);
    vec_vec_multiply(resbuf_ac, a, c, 5);

    for (int i = 0; i < 5; ++i) {
        REQUIRE( fuzzy_equals_digits(resbuf_ab[i], expect_ab[i], 4) );
        REQUIRE( fuzzy_equals_digits(resbuf_bc[i], expect_bc[i], 4) );
        REQUIRE( fuzzy_equals_digits(resbuf_ac[i], expect_ac[i], 4) );
    }
}

TEST_CASE( "row vector dotted with matrix", "[math]") {
    float v[5] = {0.3, 1.9, 2.3, 0.4, 5.2};
    float A[15] = {0.2, 0.1, 0.5,
                   1.3, 0.9, 0.3,
                   5.2, 4.5, 6.3,
                   0.6, 0.9, 0.5,
                   3.5, 9.9, 1.4};
    int M = 5, N = 3;
    float out[N];
    vec_mat_dot(out, v, A, M, N);

    float expect_out[N] = {32.9300000, 63.9300000, 22.6900000};
    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(out[i], expect_out[i], 4) );
    }
}

TEST_CASE( "matrix columns multiplied by vector element-wise", "[math]" ) {
    float A[20] = {1, 2, 3, 4,
                   4, 2, 1, 1,
                   3, 2, 2, 3,
                   1, 3, 1, 1,
                   1, 1, 1, 1};

    float v[4] = {3, 2, 2, 1};
    float resbuf[20];
    float expect[20] = { 3, 4, 6, 4,
                        12, 4, 2, 1,
                         9, 4, 4, 3,
                         3, 6, 2, 1,
                         3, 2, 2, 1};

    mat_vec_multiply(resbuf, A, 5, 4, v);
    for (int i = 0; i < 20; ++i) {
        REQUIRE( resbuf[i] == expect[i] );
    }
}

TEST_CASE( "matrix reduce sum all elements in the same row", "[math]" ) {
    float A[20] = {1, 2, 3, 4,
                   4, 2, 1, 1,
                   3, 2, 2, 3,
                   1, 3, 1, 1,
                   1, 1, 1, 1};
    float resbuf[5];
    float expected[5] = {10, 8, 10, 6, 4};
    mat_reduce_row_sum(resbuf, A, 5, 4);
    for (int i = 0; i < 5; ++i) {
        REQUIRE( resbuf[i] == expected[i] );
    }
}

TEST_CASE( "vector vector outer product", "[math]" ) {
    float a[5] = {1, 2, 3, 1, 2};
    float b[5] = {2, 3, 3, 1, 4};
    float resbuf[25];
    float expected[25] = {2, 3, 3, 1, 4,
                          4, 6, 6, 2, 8,
                          6, 9, 9, 3, 12,
                          2, 3, 3, 1, 4,
                          4, 6, 6, 2, 8};
    vec_vec_outer(resbuf, a, b, 5, 5);
    for (int i = 0; i < 25; ++i) {
        REQUIRE( resbuf[i] == expected[i] );
    }
}

TEST_CASE( "normalizing uchar to floats [0, 1]", "[math]" ) {
    unsigned char arr[10] = {4, 3, 1, 5, 1, 2, 2, 12, 4, 1};
    float resbuf[10];
    float expected[10] = {0.27272727, 0.18181818, 0, 0.36363636, 0, 0.09090909, 0.09090909, 1, 0.27272727, 0};
    normalize_ctf(resbuf, arr, 10);
    for (int i = 0; i < 10; ++i) {
        REQUIRE( resbuf[i] == expected[i] );
    }
}

TEST_CASE( "vectorized sigmoid and first derivative", "[math]" ) {
    float in[5] = {0.5, 0.12, 1.2, 0.94, 1.39};
    float out[5];
    float out_deriv[5];
    float expect_out[5] = {0.6224593312, 0.5299640518, 0.7685247835, 0.7190996574, 0.8005922432};
    float expect_out_deriv[5] = {0.2350037122, 0.2491021556, 0.1778944406, 0.2019953401, 0.1596443033};
    vec_sigmoid_and_deriv(out, out_deriv, in, 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE( fuzzy_equals_digits(out[i], expect_out[i], 6) );
        REQUIRE( fuzzy_equals_digits(out_deriv[i], expect_out_deriv[i], 6) );
    }
}

TEST_CASE( "vectorized ReLU and first derivative", "[math]" ) {
    float in[5] = {1.2, -0.3, 5.2, -2.7, 0.3};
    float out[5];
    float out_deriv[5];
    float expect_out[5] = {1.2, 0, 5.2, 0, 0.3};
    float expect_out_deriv[5] = {1, 0, 1, 0, 1};
    vec_relu_and_deriv(out, out_deriv, in, 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE( fuzzy_equals_digits(out[i], expect_out[i], 9) );
        REQUIRE( fuzzy_equals_digits(out_deriv[i], expect_out_deriv[i], 9) );
    }
}

TEST_CASE( "matrix vector multiply then reduce sum, combined operation", "[math]" ) {
    int input_nodes = 4, output_nodes = 2;
    float weights[input_nodes * output_nodes] =
        {  0.3,  0.12,
         -0.12,  0.21,
          0.38,  0.19,
          0.46, -0.05};
    float pdL_pdvals[output_nodes] = {0.51, 0.82};
    float pdL_pdout_pred[input_nodes];

    mat_vec_multiply_reduce_sum(pdL_pdout_pred, weights, input_nodes, output_nodes, pdL_pdvals);

    float expect[input_nodes] = {0.2514, 0.1110, 0.3496, 0.1936};

    for (int i = 0; i < input_nodes; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdout_pred[i], expect[i], 5) );
    }

}
