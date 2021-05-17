#include "catch.hpp"

#include "nn_layers.cu.h"
#include "math.cu.h"

TEST_CASE( "valid padding dimension calculation", "[conv2d]" ) {
    /*
        0 1 2 3 4 5 6 7 8
        --1--
            --2--
            --3--
                --4--
                --5-- 
                    --6--
                    --7--
        (9 - 3) / (int)1 + 1 = 6 / (int)1 + 1 = 7
     */
    REQUIRE( calc_dims_pad_valid(9, 3, 1) == 7 );

    /*
       0 1 2 3 4 5 6 7 8 9 a b c d e
       --1--
             --2--
                   --3--
                         --4--
                               --5--
       (15 - 3) / (int)3 + 1 = 12  / (int)3 + 1 = 5
     */
    
    REQUIRE( calc_dims_pad_valid(15, 3, 3) == 5 );
    /*
        00-04 02-06 04-08 06-10 08-12 10-14
        12-16 14-18 16-20 18-22 20-24 22-26
        24-28 26-30 28-32 30-34 32-36 34-38
        36-40 38-42 40-44 42-46 44-48 46-50
        48-52 50-54 52-56 54-58 56-60 58-62
     */
    REQUIRE( calc_dims_pad_valid(64, 5, 2) == 30 );
}

TEST_CASE( "convolution layer forward pass", "[conv2d]" ) {
    int i_rows = 4, i_cols = 4;
    float ins[i_rows * i_cols] = 
        {.3, .2, .3,  .6,
         .2, .1, .5,  .9,
         .1, .2, .7, 1.0,
          0, .2, .4,  .8};
    int w_rows = 2, w_cols = 2;
    int s_rows = 2, s_cols = 2, filters = 2;
    float weights[filters * w_rows * w_cols] =
        { 1.0,  0.2,
         -0.2, -1.0,

         -0.3,  0.5,
          0.6,  0.7};
    // input 4 rows, kernel 2 rows, stride 2 rows
    int o_rows = calc_dims_pad_valid(i_rows, w_rows, s_rows);
    // input 4 cols, kernel 2 cols, stride 2 cols
    int o_cols = calc_dims_pad_valid(i_cols, w_cols, s_cols);
    float outs[filters * o_rows * o_cols];
    float vals[filters * o_rows * o_cols];
    float do_dv[filters * o_rows * o_cols];
    
    float *devIns, *devWeights, *devOuts, *devVals, *devDoDv;
    cudaMalloc(&devIns, i_rows * i_cols * sizeof(float));
    cudaMalloc(&devWeights, filters * w_rows * w_cols * sizeof(float));
    cudaMalloc(&devOuts, filters * o_rows * o_cols * sizeof(float));
    cudaMalloc(&devVals, filters * o_rows * o_cols * sizeof(float));
    cudaMalloc(&devDoDv, filters * o_rows * o_cols * sizeof(float));
    cudaMemcpy(devIns, ins, i_rows * i_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devWeights, weights, filters * w_rows * w_cols * sizeof(float), cudaMemcpyHostToDevice);

    Conv2D_forward(devOuts, o_rows, o_cols, devVals, devDoDv,
                   devIns, i_rows, i_cols,
                   devWeights, w_rows, w_cols,
                   s_rows, s_cols, filters,
                   0); // 0 == sigmoid activation
    
    cudaMemcpy(vals, devVals, filters * o_rows * o_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outs, devOuts, filters * o_rows * o_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(do_dv, devDoDv, filters * o_rows * o_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devIns); cudaFree(devWeights); cudaFree(devOuts); cudaFree(devVals); cudaFree(devDoDv);

    float expect_vals[filters * o_rows * o_cols] =
        { 0.2000000000, -0.5800000000,
         -0.0600000000,  0.0200000000,
        
          0.2000000000, 1.14000000000,
          0.2100000000, 1.09000000000};
    float expect_outs[filters *o_rows * o_cols] =
        {0.5498339973, 0.3589325937,
         0.4850044984, 0.5049998333,
        
         0.5498339973,  0.7576796390,
         0.5523079096,  0.7483817216};
    float expect_do_dv[filters * o_rows * o_cols] =
        {0.2475165727, 0.2300999869,
         0.2497751349, 0.2499750017,
        
         0.2475165727, 0.1836012036,
         0.2472638826, 0.1883065204};

    for (int i = 0; i < filters * o_rows * o_cols; ++i) {
        //printf("vals[%d]=%f, expect %f\n", i, vals[i], expect_vals[i]);
        REQUIRE( fuzzy_equals_digits(vals[i], expect_vals[i], 6) );
    }
    for (int i = 0; i < filters * o_rows * o_cols; ++i) {
        REQUIRE( fuzzy_equals_digits(outs[i], expect_outs[i], 6) );
    }
    for (int i = 0; i < filters * o_rows * o_cols; ++i) {
        REQUIRE( fuzzy_equals_digits(do_dv[i], expect_do_dv[i], 6) );
    }
}

TEST_CASE( "convolution layer backprop", "[conv2d]" ) {
    int o_rows = 2, o_cols = 2; // output dims during forward-feed
    int w_rows = 2, w_cols = 2; // kernel dims
    int i_rows = 4, i_cols = 4; // input dims during forward-feed
    int s_rows = 2, s_cols = 2, filters = 1; // stride and number of filters

    float pdL_pdouts[o_rows * o_cols] =
        { 2.3, -1.2,
         -0.4, 1.2 };
    float douts_dvals[o_rows * o_cols] =
        {0.2475165727, 0.2300999869,
         0.2497751349, 0.2499750017};
    float weights[w_rows * w_cols] =
        {0.42, -0.12,
         2.12, -1.23};
    float ins[i_rows * i_cols] = 
        {1, 2, 2, 1,
         2, 2, 1, 1,
         1, 1, 2, 2,
         1, 3, 1, 3};

    float pdL_pdvals[o_rows * o_cols];
    float pdL_pdouts_pred[i_rows * i_cols];
    float grads[w_rows * w_cols];

    float *devPdL_pdouts, *devDouts_dvals, *devWeights, *devIns, *devPdL_pdvals, *devPdL_pdouts_pred, *devGrads;
    cudaMalloc(&devPdL_pdouts, o_rows * o_cols * sizeof(float));
    cudaMalloc(&devDouts_dvals, o_rows * o_cols * sizeof(float));
    cudaMalloc(&devWeights, w_rows * w_cols * sizeof(float));
    cudaMalloc(&devIns, i_rows * i_cols * sizeof(float));
    cudaMalloc(&devPdL_pdvals, o_rows * o_cols * sizeof(float));
    cudaMalloc(&devPdL_pdouts_pred, i_rows * i_cols * sizeof(float));
    cudaMalloc(&devGrads, w_rows * w_cols * sizeof(float));

    cudaMemcpy(devPdL_pdouts, pdL_pdouts, o_rows * o_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devDouts_dvals, douts_dvals, o_rows * o_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devWeights, weights, w_rows * w_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devIns, ins, i_rows * i_cols * sizeof(float), cudaMemcpyHostToDevice);

    Conv2D_backward(
        o_rows, o_cols,
        devPdL_pdouts, devDouts_dvals, devPdL_pdvals, devPdL_pdouts_pred,
        devIns, i_rows, i_cols,
        devWeights, w_rows, w_cols,
        devGrads,
        s_rows, s_cols, filters);

    cudaMemcpy(pdL_pdvals, devPdL_pdvals, o_rows * o_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pdL_pdouts_pred, devPdL_pdouts_pred, i_rows * i_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grads, devGrads, w_rows * w_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devPdL_pdouts); cudaFree(devDouts_dvals); cudaFree(devWeights);
    cudaFree(devIns); cudaFree(devPdL_pdvals); cudaFree(devPdL_pdouts_pred);
    cudaFree(devGrads);

    float expect_pdL_pdvals[o_rows * o_cols] = 
        { 0.56928811720, -0.2761199843,
         -0.09991005396,  0.2999700020};
    
    // skipped because not implemented yet. need to implement padding and dilation
    //float expect_pdL_pdouts_pred[i_rows * i_cols];

    float expect_grads[w_rows * w_cols] =
        { 0.5170780986, 1.3624862001,
          1.0625161981, 1.4626360942};

    for (int i = 0 ; i < o_rows * o_cols; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdvals[i], expect_pdL_pdvals[i], 6) );
    }
    for (int i = 0; i < w_rows * w_cols; ++i) {
        //printf("grads[%d]=%f, expect %f\n", i, grads[i], expect_grads[i]);
        REQUIRE( fuzzy_equals_digits(grads[i], expect_grads[i], 6) );
    }
}

TEST_CASE( "dense layer forward pass", "[dense]" ) {
    int i_len = 5, o_len = 2;
    float ins[i_len] = {3.43, 5.3, 1.02, 0.43, 2.4};
    float weights[i_len * o_len] =
        { 0.34,  0.93,
         -0.03,  0.12,
          0.13, -0.42,
          0.94, -0.29,
         -0.07,  0.52};
    float vals[o_len];
    float outs[o_len];
    float do_dv[o_len];
   
    float *devIns, *devWeights, *devVals, *devOuts, *devDoDv;
    cudaMalloc(&devIns, i_len * sizeof(float));
    cudaMalloc(&devWeights, i_len * o_len * sizeof(float));
    cudaMalloc(&devVals, o_len * sizeof(float));
    cudaMalloc(&devOuts, o_len * sizeof(float));
    cudaMalloc(&devDoDv, o_len * sizeof(float));
    cudaMemcpy(devIns, ins, i_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devWeights, weights, i_len * o_len * sizeof(float), cudaMemcpyHostToDevice);

    Dense_forward(
        devOuts,o_len,
        devVals, devDoDv,
        devIns, i_len,
        devWeights,
        0 // sigmoid activation
    );
    
    cudaMemcpy(vals, devVals, o_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outs, devOuts, o_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(do_dv, devDoDv, o_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devIns); cudaFree(devWeights); cudaFree(devVals);
    cudaFree(devOuts); cudaFree(devDoDv);

    float expect_vals[o_len] = {1.3760000000, 4.5208000000};
    float expect_outs[o_len] = {0.7983478144, 0.9892367912};
    float expect_do_dv[o_len] = {0.1609885816, 0.01064736214};
    for (int i = 0; i < o_len; ++i) {
        REQUIRE( fuzzy_equals_digits(vals[i], expect_vals[i], 5) );
    }
    for (int i = 0; i < o_len; ++i) {
        REQUIRE( fuzzy_equals_digits(outs[i], expect_outs[i], 6) );
    }
    for (int i = 0; i < o_len; ++i) {
        REQUIRE( fuzzy_equals_digits(do_dv[i], expect_do_dv[i], 5) );
    }
}

TEST_CASE( "dense layer backward pass", "[dense]" ) {
    int i_len = 4, o_len = 2, activation = 0;
    float ins[i_len] = {1.3, 0.3, -0.4, 0.8};
    float weights[i_len * o_len] = 
        {0.3, 0.5,
         0.1, 0.2,
         0.4, 0.1,
         0.2, 0.3};
    float pdL_pdouts[o_len] = {0.08, 0.12};
    float douts_dvals[o_len] = {0.8, 0.9};
    float pdL_pdvals[o_len];
    float pdL_pdouts_pred[i_len];
    float grads[i_len * o_len];

    float *devIns, *devWeights, *devPdL_pdouts, *devDouts_dvals, *devPdL_pdvals, *devPdL_pdouts_pred, *devGrads;
    cudaMalloc(&devIns, i_len * sizeof(float));
    cudaMalloc(&devWeights, i_len * o_len * sizeof(float));
    cudaMalloc(&devPdL_pdouts, o_len * sizeof(float));
    cudaMalloc(&devDouts_dvals, o_len * sizeof(float));
    cudaMalloc(&devPdL_pdvals, o_len * sizeof(float));
    cudaMalloc(&devPdL_pdouts_pred, i_len * sizeof(float));
    cudaMalloc(&devGrads, i_len * o_len * sizeof(float));

    cudaMemcpy(devIns, ins, i_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devWeights, weights, i_len * o_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devPdL_pdouts, pdL_pdouts, o_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devDouts_dvals, douts_dvals, o_len * sizeof(float), cudaMemcpyHostToDevice);

    Dense_backward(
        o_len, devPdL_pdouts, devDouts_dvals, devPdL_pdvals, devPdL_pdouts_pred,
        devIns, i_len, devWeights, devGrads, activation
    );

    cudaMemcpy(pdL_pdvals, devPdL_pdvals, o_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pdL_pdouts_pred, devPdL_pdouts_pred, i_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(grads, devGrads, i_len * o_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devIns); cudaFree(devWeights); cudaFree(devPdL_pdouts); cudaFree(devDouts_dvals);
    cudaFree(devPdL_pdvals); cudaFree(devPdL_pdouts_pred); cudaFree(devGrads);

    float expect_pdL_pdvals[o_len] = {0.064, 0.108};
    float expect_pdL_pdouts_pred[i_len] = {0.0732, 0.028, 0.0364, 0.0452};
    float expect_grads[i_len * o_len] = 
        { 0.0832, 0.1404,
          0.0192, 0.0324,
         -0.0256,-0.0432,
          0.0512, 0.0864};
    for (int i = 0; i < o_len; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdvals[i], expect_pdL_pdvals[i], 5) );
    }
    for (int i = 0; i < i_len; ++i) {
        REQUIRE( fuzzy_equals_digits(pdL_pdouts_pred[i], expect_pdL_pdouts_pred[i], 5) );
    }
    for (int i = 0; i < i_len * o_len; ++i) {
        REQUIRE( fuzzy_equals_digits(grads[i], expect_grads[i], 5) );
    }
}


TEST_CASE( "stochastic gradient descent update params", "[dense][conv2d]" ) {
    float eta = 0.5;
    int N = 4;
    float weights[N] = {0.12, -0.56, 0.39, 0.93};
    float grads[N] = {0.2, -0.3, 0.1, 0.3};
    
    float *devWeights, *devGrads;
    cudaMalloc(&devWeights, N * sizeof(float));
    cudaMalloc(&devGrads, N * sizeof(float));
    cudaMemcpy(devWeights, weights, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devGrads, grads, N * sizeof(float), cudaMemcpyHostToDevice);
    SGD_update_params(eta, devWeights, devGrads, N);
    cudaMemcpy(weights, devWeights, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(devWeights); cudaFree(devGrads);
    float expect_weights[N] = {0.02, -0.41, 0.34, 0.78};
    for (int i = 0; i < N; ++i) {
        REQUIRE( fuzzy_equals_digits(weights[i], expect_weights[i], 5) );
    }
}




