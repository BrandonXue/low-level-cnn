#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <wchar.h>

__host__
unsigned char *load_img_carefully(const char*, int, int, int, int*);

__host__
wchar_t int256_to_unicode(int);

__host__
void img_to_ascii(unsigned char*, int, int, int);

#endif
