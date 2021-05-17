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

TEST_CASE( "vector argmax", "[tensor]" ) {
    float input[5] = {-0.1359, 0.0943492, 0.281114, -0.9814, 0.68318};
    int answer = argmax(input, 5);
    int expect = 4;
    REQUIRE(expect == answer);
}

TEST_CASE( "vector and vector element-wise multiplication", "[math]" ) {
    float a[5] = {2.1, 1.8, 1.2, 0.4, 0.8};
    float b[5] = {1.0, 1.1, 2.4, 3.1, 1.9};
    float resbuf_ab[5];
    float expect_ab[5] = {2.1, 1.98, 2.88, 1.24, 1.52};
    float *devA, *devB, *devRes;
    cudaMalloc(&devA, 5 * sizeof(float));
    cudaMalloc(&devB, 5 * sizeof(float));
    cudaMalloc(&devRes, 5 * sizeof(float));
    cudaMemcpy(devA, a, 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, 5 * sizeof(float), cudaMemcpyHostToDevice);

    vec_vec_multiply(devRes, devA, devB, 5);

    cudaMemcpy(resbuf_ab, devRes, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devA); cudaFree(devB); cudaFree(devRes);

    for (int i = 0; i < 5; ++i) {
        REQUIRE( fuzzy_equals_digits(resbuf_ab[i], expect_ab[i], 4) );
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
    float expect_out[N] = {32.9300000, 63.9300000, 22.6900000};

    float *devOut, *devA, *devV;
    cudaMalloc(&devOut, N * sizeof(float));
    cudaMalloc(&devV, M * sizeof(float));
    cudaMalloc(&devA, M * N * sizeof(float));
    cudaMemcpy(devV, v, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devA, A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    vec_mat_dot(devOut, devV, devA, M, N);

    cudaMemcpy(out, devOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devOut); cudaFree(devV); cudaFree(devA);

    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(out[i], expect_out[i], 4) );
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
    float *devA, *devB, *devRes;
    cudaMalloc(&devA, 5 * sizeof(float));
    cudaMalloc(&devB, 5 * sizeof(float));
    cudaMalloc(&devRes, 25 * sizeof(float));
    cudaMemcpy(devA, a, 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    vec_vec_outer(devRes, devA, devB, 5, 5);
    
    cudaMemcpy(resbuf, devRes, 25 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devA); cudaFree(devB); cudaFree(devRes);

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
    
    float *devIn, *devOut, *devOutDeriv;
    cudaMalloc(&devIn, 5 * sizeof(float));
    cudaMalloc(&devOut, 5 * sizeof(float));
    cudaMalloc(&devOutDeriv, 5 * sizeof(float));
    cudaMemcpy(devIn, in, 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    vec_sigmoid_and_deriv(devOut, devOutDeriv, devIn, 5);
    
    cudaMemcpy(out, devOut, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_deriv, devOutDeriv, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devIn); cudaFree(devOut); cudaFree(devOutDeriv);

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
    
    float *devIn, *devOut, *devOutDeriv;
    cudaMalloc(&devIn, 5 * sizeof(float));
    cudaMalloc(&devOut, 5 * sizeof(float));
    cudaMalloc(&devOutDeriv, 5 * sizeof(float));
    cudaMemcpy(devIn, in, 5 * sizeof(float), cudaMemcpyHostToDevice);
    
    vec_relu_and_deriv(devOut, devOutDeriv, devIn, 5);
    
    cudaMemcpy(out, devOut, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_deriv, devOutDeriv, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devIn); cudaFree(devOut); cudaFree(devOutDeriv);

    for (int i = 0; i < 5; ++i) {
        REQUIRE( fuzzy_equals_digits(out[i], expect_out[i], 9) );
        REQUIRE( fuzzy_equals_digits(out_deriv[i], expect_out_deriv[i], 9) );
    }
}

TEST_CASE( "matrix dotted with column vector", "[math]" ) {
    int input_nodes = 4, output_nodes = 2;
    float weights[input_nodes * output_nodes] =
        {  0.3,  0.12,
         -0.12,  0.21,
          0.38,  0.19,
          0.46, -0.05};
    float pdL_pdvals[output_nodes] = {0.51, 0.82};
    float pdL_pdout_pred[input_nodes];

    float *devVec, *devMat, *devRes;
    cudaMalloc(&devVec, output_nodes * sizeof(float));
    cudaMalloc(&devMat, input_nodes * output_nodes * sizeof(float));
    cudaMalloc(&devRes, input_nodes * sizeof(float));
    cudaMemcpy(devMat, weights, input_nodes * output_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devVec, pdL_pdvals, output_nodes * sizeof(float), cudaMemcpyHostToDevice); 
    
    mat_vec_dot(devRes, devMat, input_nodes, output_nodes, devVec);
    
    cudaMemcpy(pdL_pdout_pred, devRes, input_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devVec); cudaFree(devMat); cudaFree(devRes);

    float expect[input_nodes] = {0.2514, 0.1110, 0.3496, 0.1936};

    for (int i = 0; i < input_nodes; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdout_pred[i], expect[i], 5) );
    }

}

TEST_CASE( "mse works", "[loss_func]" ) {
    int N = 6;
    float y_true[N] = {12.2, 19.3, 94.2, 4.5, 8.39, 28.39};
    float y_pred[N] = {14.3, 20.3, 91.3, 4.9, 9.01, 25.93};
    REQUIRE( fuzzy_equals_digits(mse(y_true, y_pred, N), 3.402666666, 4));
}

TEST_CASE( "cross entropy works", "[loss_func]" ) {
    // cat_cross_entropy(int len, float *y_true, float *y_pred, float *pdL_pdval, float *loss)
    int N = 4;
    float y_true[N] = {0, 1, 0, 0};
    float vals[N] = {1.2, 3.4, 1.1, 2.4};
    float outs[N];
    float pdL_pdval[N];
    float loss;

    float expect_outs[N] = {0.07017559697, 0.63333571, 0.06349750597, 0.2329911871};

    // These answers were calculated by hand dotting the Jacobian and the ∂L/∂outs,
    // instead of softmax(vals) - y_true
    // if they end up the same, then the math should be correct
    float expect_pdL_pdval[N] = {0.07017322697, -0.3666519068, 0.06349536151, 0.2329833184};

    cat_cross_entropy(N, y_true, vals, outs, pdL_pdval, &loss);
    
    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(outs[i], expect_outs[i], 5) );
    }

    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdval[i], expect_pdL_pdval[i], 4) );
    }


}

TEST_CASE( "one single convolution operation", "[math]" ) {
    float image[16] =
    {1, 2, 3, 4,
     2, 1, 3, 3,
     1, 1, 2, 2,
     3, 2, 1, 1};
    float kernel[4] =
    {0.5, 1.0,
     2.0, 0.5};
    float res =
        convolution_2d(1, 0, // output row 1, col 0. the bottom left
                        image , 4, 4,
                        kernel, 2, 2,
                        2, 2);
    REQUIRE( fuzzy_equals_digits(res, 8.5, 8) );
    
}

TEST_CASE( "convolution operation on a whole matrix", "[math]" ) {
    float image[16] =
    {1, 2, 3, 4,
     2, 1, 3, 3,
     1, 1, 2, 2,
     3, 2, 1, 1};
    float kernel[4] =
    {0.5, 1.0,
     2.0, 0.5};
    float res[4];
    float expect[4] =
    {7.0, 13.0,
     8.5, 5.5};

    float *devImage, *devKernel, *devRes;
    cudaMalloc(&devImage, 16 * sizeof(float));
    cudaMalloc(&devKernel, 4 * sizeof(float));
    cudaMalloc(&devRes, 4 * sizeof(float));
    cudaMemcpy(devImage, image, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devKernel, kernel, 4 * sizeof(float), cudaMemcpyHostToDevice);

    all_convolution_2d(
        devRes,
        devImage, 4, 4,
        devKernel, 1, 2, 2,
        2, 2);
    
    cudaMemcpy(res, devRes, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devImage); cudaFree(devKernel); cudaFree(devRes);

    for (int i = 0; i < 4; ++i) {
        //printf("res[%d]=%f, expect %f\n", i, res[i], expect[i]);
        REQUIRE( fuzzy_equals_digits(res[i], expect[i], 8) );
    }

}
