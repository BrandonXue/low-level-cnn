#ifndef IMAGE_IO_H
#define IMAGE_IO_H

__host__
unsigned char *load_img_carefully(const char*, int, int, int, int*);

__host__
char int256_to_ascii(int);

__host__
void img_to_ascii(unsigned char*, int, int, int);

#endif
