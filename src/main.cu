// Standard
#include <stdio.h>

// Third-party
#include <cuda.h>

// Local
#include "conv2d.cu.h"
#include "image_io.cu.h"
#include "string.cu.h"

enum Command{EXIT, HELP, SHOW, UNKNOWN};

Command get_command(Tokens *toks) {
    if (Tokens_match_at(toks, 0, "exit") ||
        Tokens_match_at(toks, 0, "quit")) {
        return EXIT;
    }
    if (Tokens_match_at(toks, 0, "help") ||
        Tokens_match_at(toks, 0, "?") ||
        Tokens_match_at(toks, 0, "h")) {
        return HELP;
    }
    if (Tokens_match_at(toks, 0, "show")) {
        return SHOW;
    }
    return UNKNOWN;
}

void print_prompt() {
    printf("p2-sh> ");
}

void cmd_help() {
    printf("p2-sh> \n");
    printf("\tCommands:\n"
           "\texit:\tQuit the program. Model does not save automatically.\n"

           "\thelp:\tPrint this help.\n"

           "\tshow:\n"
               "\t\timage <filename>: Print an image as ASCII art.\n"
               "\t\tDo not need to type the path in filename.\n"); 
}

void cmd_show(Tokens *toks) {
    if (Tokens_match_at(toks, 1, "image")) {
        if (Tokens_get_count(toks) < 3) {
            printf("Error: Missing last argument for \"show image\".\n");
            return;
        }
        int status;
        char buf[128];
        char prefix[] = "data/chinese_mnist/data/";
        strcpy(buf, prefix);
        strcat(buf, Tokens_at(toks, 2));
        unsigned char *image = load_img_carefully(
            buf, 64, 64, 1, &status
        );
        //set_locale_lc_all();
        if (status == 0) {
            img_to_ascii(image, 64, 64, 1);
            free(image);
        } else {
            printf("Error: Image could not be loaded: %s\n", buf);
        }
    } 
    else {
        printf("Error: \"show\" must be used with a valid sub-command.\n");
    }
}

void interactive_loop() {
    
    char buf[1024];
    while (true) {
        print_prompt();
        fgets(buf, 1023, stdin);
            
        Tokens *toks = Tokens_create();
        
        // If tokens not fetched successfully
        if (!Tokens_fetch(toks, buf)) {
            printf("Could not detect the tokens in your command.\n");
            continue;
        }
        
        Command cmd = get_command(toks);
        
        // If command not understood
        if (cmd == UNKNOWN) {
            printf("Could not understand the command.\n");
            continue;
        }
        
        /** Handle commands **/

        if (cmd == HELP) {
            cmd_help();
        }

        else if (cmd == EXIT) {
            break;
        }

        else if (cmd == SHOW) {
            cmd_show(toks);
        }

        toks = Tokens_destroy(toks);
    }
}


int main(int argc, char *argv[]) {

    /*int status;
    unsigned char *data = load_img_carefully(
        "data/chinese_mnist/data/input_1_1_1.jpg",
        64, 64, 1, &status
    );*/
    //set_locale_lc_all();
    //img_to_unicode(data, 64, 64, 1);
    /*double weights_k1[3] = {-479, 479, 999};
    Double2D kern1 = {.width = 5, .height = 5, .data = weights_k1};
    UChar2D img = {.width = 64, .height = 64, .data = data};
    Double2D out = {.width = 0, .height = 0};
    conv_2d_input(&kern1, 1, 1, PAD_VALID, &img, &out);
    printf("Output dimensions: %d, %d.\n", out.width, out.height);
    */
    InputLabel *inputs = load_chinese_mnist_info();
    /*for (int i = 0; i < 14999; ++i) {
        printf("Input file %s has label %d\n", inputs[i].input, inputs[i].label);
    }*/
    interactive_loop();
    return 0;
}
