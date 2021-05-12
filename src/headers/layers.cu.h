#ifndef CONV_2D_CU_H
#define CONV_2D_CU_H

enum PadType{PAD_VALID, PAD_SAME_ZEROS};

struct UChar2D {
    int width;
    int height;
    unsigned char *data;
};

struct Double2D {
    int width;
    int height;
    double *data;
};

__host__
void conv_2d(Double2D*, int, int, PadType, Double2D*, Double2D*);

__host__
void conv_2d_input(Double2D*, int, int, PadType, UChar2D*, Double2D*);

__host__
int calc_dims_pad_valid(int, int, int);

__host__
void Conv2D_forward(
    float *outs, int o_rows, int o_cols, // output
    float *vals, // the values before activation; same dimensions as outs
    float *do_dv, // change in outs w.r.t. vals; same sims as outs
    float *ins, int i_rows, int i_cols, // input
    float *weights, int w_rows, int w_cols, // the filters
    int s_rows, int s_cols, int filters, // stride and filters
    int activation // 0 = sigmoid, 1 = ReLU
);

#endif
