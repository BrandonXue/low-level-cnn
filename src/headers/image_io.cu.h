#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <wchar.h>

struct InputLabel {
    char *input;
    int label;
};

__host__
unsigned char *load_img_carefully(const char*, int, int, int, int*);

__host__
void set_locale_lc_all();

__host__
char int256_to_ascii(int);

__host__
void img_to_ascii(unsigned char*, int, int, int);

__host__
wchar_t int256_to_unicode(int);

__host__
void img_to_unicode(unsigned char*, int, int, int);

__host__
InputLabel *load_chinese_mnist_info();

#endif
