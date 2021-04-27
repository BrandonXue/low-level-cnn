#include <cuda.h>
#include "catch.hpp"

__host__ int myfunc(int in) {
    return -in;
}

int main(int argc, char *argv[]) {
    return 0;
}
