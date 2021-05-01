#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image_io.cu.h"

/**
 * Load an image (preferably jpeg) using stbi_load().
 * Warns when the actual properties of the loaded image differ from
 * the expected properties.
 *
 * On success, returns a pointer to the image array and sets status to 0.
 * On failure, returns NULL and sets status to nonzero. The bytes of the
 * nonzero return mean the following:
 *     0b1000 (8): Unknown error. stbi_load() returned NULL.
 *     0b0100 (4): Width differs from expect_wid
 *     0b0010 (2): Height differs from expect_hgt
 *     0b0001 (1): Components/channels differs from expect_c 
 *
 * @return A pointer to the image array, with components interleaved.
 *     Or returns NULL if something went wrong.
 */
unsigned char *load_img_carefully(
    const char* file_path,
    int expect_wid, int expect_hgt, int expect_c,
    int *status
) {
    int width, height, components;
    unsigned char *data = stbi_load(
        file_path, &width, &height, &components, 0
    );
    
    // Clear status
    *status = 0;

    // If fail loading
    if (!data) {
        *status = *status & 8;
    }
    // On load succeed
    else {
        if (expect_wid != width) {
            *status &= 4;
        }
        if (expect_hgt != height) {
            *status &= 2;
        }
        if (expect_c != components) {
            *status &= 1;
        }
    }
    return data;
}


