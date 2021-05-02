// Standard
#include <locale.h>
#include <stdio.h>
#include <wchar.h>

// Third party
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include "stb_image.h"

// Local
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
__host__
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
    if (data == NULL) {
        *status |= 8;
    }
    // On load succeed
    else {
        if (expect_wid != width) {
            *status |= 4;
        }
        if (expect_hgt != height) {
            *status |= 2;
        }
        if (expect_c != components) {
            *status |= 1;
        }
    }
    return data;
}

/**
 * Convert single-byte numeric intensity into an ASCII character,
 * where the character chosen is visually darker the higher the
 * intensity is.
 */
__host__
wchar_t int256_to_unicode(int intensity) {
    if (intensity >= 192)   // 256 * 3/4
        return (wchar_t)0x2593; // dark shade
    if (intensity >= 128)   // 256 * 2/4
        return (wchar_t)0x2592; // medium shade
    if (intensity >= 64)    // 256 * 1/4
        return (wchar_t)0x2591; // light shade
                            // 256 * 0/4
    return (wchar_t)0x0020;     // whitespace
}

__host__
char int256_to_ascii(int intensity) {
    if (intensity >= 204) // 256 * 4/5
        return '#';
    if (intensity >= 153) // 256 * 3/5
        return '*';
    if (intensity >= 102) // 256 * 2/5
        return '+';
    if (intensity >= 51)  // 256 * 1/5
        return '.';
    return ' ';
}

/**
 * Print 1 channel grayscale or 3 channel RGB image as ASCII art.
 */
__host__
void img_to_ascii(unsigned char *data, int width, int height, int c) {
    if (data != NULL && (c == 1 || c == 3)) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int avg = 0;
                for (int chn = 0; chn < c; ++chn) {
                    // y * width + x gives the pixel, multiply it by channels
                    // add current channel: chn
                    avg += (int)data[c * (y * width + x) + chn];
                }
                avg /= c;
                printf("%c", int256_to_ascii(avg));
            }
            printf("\n");
        }
    }
}

/**
 * Must be called before attempting to print Unicode.
 */
__host__
void set_locale_lc_all() {
    setlocale(LC_ALL, "");
}

/**
 * Print 1 channel grayscale or 3 channel RGB image as Unicode art.
 */
__host__
void img_to_unicode(unsigned char *data, int width, int height, int c) {
    if (data != NULL && (c == 1 || c == 3)) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int avg = 0;
                for (int chn = 0; chn < c; ++chn) {
                    // y * width + x gives the pixel, multiply it by channels
                    // add current channel: chn
                    avg += (int)data[c * (y * width + x) + chn];
                }
                avg /= c;
                printf("%lc", (wchar_t)int256_to_unicode(avg));
            }
            printf("\n");
        }
    }
}


