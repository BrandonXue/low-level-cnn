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

#endif
