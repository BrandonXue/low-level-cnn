#include <cuda.h>

#include <stdio.h>

#include "placeholder.cu.h"
#include "catch.hpp"

int main(int argc, char *argv[]) {
    printf("Hello from program main!\n");
    printf("Testing included cuda header: %d.\n", (int)placeholder());
    return 0;
}
