// Standard
#include <stdio.h>

// Third-party
#include <cuda.h>

// Local
#include "cn_mnist_utils.cu.h"
#include "nn_layers.cu.h"
#include "image_io.cu.h"
#include "math.cu.h"
#include "string.cu.h"

enum Command{EXIT, HELP, PREDICT, PREPROCESS, SHOW, TRAIN, UNKNOWN};

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
    if (Tokens_match_at(toks, 0 , "predict")) {
        return PREDICT;
    }
    if (Tokens_match_at(toks, 0, "preprocess")) {
        return PREPROCESS;
    }
    if (Tokens_match_at(toks, 0, "show")) {
        return SHOW;
    }
    if (Tokens_match_at(toks, 0, "train")) {
        return TRAIN;
    }
    return UNKNOWN;
}

void print_prompt() {
    printf("p2-sh> ");
}

void cmd_help() {
    printf("\tCommands:\n\n"

           "\texit\n"
               "\t\tQuit the program. Model does not save automatically.\n\n"

           "\thelp\n"
               "\t\tPrint this help.\n\n"

           "\tpreprocess\n"
               "\t\tLoad the Chinese MNIST metadata and preprocess images.\n\n"

           "\tshow (image <filename>) | (preprocessed <index>)\n"
               "\t\tshow image <filename>: Print an image as ASCII art.\n"
               "\t\tDo not need to type the path in filename.\n"
               "\t\tshow preprocessed <index>: Same as show image except\n"
               "\t\tthis uses the pre-processed data. This is mainly just\n"
               "\t\ta sanity check.\n\n"
            
           "\ttrain <n-iters> [--load [--save]]\n"
               "\t\tTrain the neural network for the specified number of inputs.\n"
               "\t\tThis does NOT do a train-test split. All data may be used.\n"
               "\t\tOptionally specify whether weights should be saved.\n\n"
    );
}

void cmd_show(Tokens *toks, float *data) {
    if (Tokens_match_at(toks, 1, "image")) {
        if (Tokens_get_count(toks) < 3) {
            printf("Error: Missing file name for \"show image\".\n");
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
    } else if (Tokens_match_at(toks, 1, "preprocessed")) {
        if (data == NULL) {
            printf("Error: Preprocess the data first.\n");
        } 
        if (Tokens_get_count(toks) < 3) {
            printf("Error: Missing the index of the image for \"show preprocessed\".\n");
            return;
        }
        char *index = Tokens_at(toks, 2);
        if ( (atoi(index) > 0 || Tokens_match_at(toks, 2, "0")) && atoi(index) < 14999 ) {
            flt_img_to_ascii(data + atoi(index) * 64 * 64, 64, 64, 1); 
        } else {
            printf("Error: Enter an index between [0, 14998]\n");
        }
    } else {
        printf("Error: \"show\" must be used with a valid sub-command.\n");
    }
}

void cmd_predict(Tokens *toks) {
    if (Tokens_get_count(toks) < 2) {
        printf("Error: Specify an image file to predict.\n");
        return;
    }
    int status;
    char buf[128];
    char prefix[] = "data/chinese_mnist/data/";
    strcpy(buf, prefix);
    strcat(buf, Tokens_at(toks, 1));
    unsigned char *image = load_img_carefully(
        buf, 64, 64, 1, &status
    );
    if (status == 0) {
        // process and predict
        float *image_flt = (float*)malloc(64 * 64 * sizeof(float));
        for (int pixel = 0; pixel < 64 * 64; ++pixel) {
            image_flt[pixel] = (float)image[pixel] / 255.0;
        }
        flt_img_to_ascii(image_flt, 64, 64, 1);
        free(image_flt);
    } else {
        printf("Error: Image could not be loaded: %s\n", buf);
    }
    free(image);
}

void cmd_preprocess(InputLabel **metadata, float **data) {
    const char *filepath = "data/chinese_mnist/processed.data";
    
    // If already loaded, don't do anything
    if (*data != NULL) {
        printf("Metadata and data already loaded.\n");
        return;
    }

    printf("Loading Chinese MNIST metadata. (This contains filenames and target labels.)\n");
    *metadata = load_chinese_mnist_info();
    *data = (float*)malloc(64 * 64 * 14999 * sizeof(float));

    FILE *processed_imgs_file = fopen(filepath, "rb");
    if (processed_imgs_file == NULL) {
        printf("Did not find previously pre-processed images. Loading and pre-processing now.\n");
        preprocess_images(*data, *metadata);
        
        printf("Saving to file %s\n\n", filepath);
        processed_imgs_file = fopen(filepath, "wb");
        fwrite(*data, sizeof(float), 64 * 64 * 14999, processed_imgs_file);
        fclose(processed_imgs_file);
    } else {
        printf("Found previously pre-processed images. Loading now.\n");
        fread(*data, sizeof(float), 64 * 64 * 14999, processed_imgs_file);
        fclose(processed_imgs_file);
    }
}

int cmd_train(InputLabel *metadata, float *data, Tokens *toks) {
    if (metadata == NULL || data == NULL) {
        printf("Error: Preprocess the images first. (This will also load the metadata.)\n");
        return 0;
    } 
    char *iters = Tokens_at(toks, 1);
    if (iters == NULL || atoi(iters) <= 0) {
        printf("Error: Enter a valid number of iterations.\n");
        return 0;
    } else {
        printf("Model will train for %d iterations.\n", atoi(iters));
    }
    bool load = Tokens_match_at(toks, 2, "--load");
    bool save = false;
    if (load) {
        save = Tokens_match_at(toks, 3, "--save");
        printf("Model will be saved at the end of training.\n");
    }
    
    /**** Setup configuration and allocate memory for the model ****/
    
    /** Layer 1: Conv2D Layer **/

    // CONFIGURABLE
    int l1_in_rows = 64, l1_in_cols = 64;
    int l1_filters = 16;
    int l1_kernel_rows = 5, l1_kernel_cols = 5;
    int l1_stride_rows = 3, l1_stride_cols = 3;
    int l1_activation = 1; // the code for ReLU
    
    // Output dimensions depend on input dimensions, kernel size, and stride
    int l1_out_rows = calc_dims_pad_valid(l1_in_rows, l1_kernel_rows, l1_stride_rows);
    int l1_out_cols = calc_dims_pad_valid(l1_in_cols, l1_kernel_cols, l1_stride_cols);
    
    // Each filter has (kernel_rows * kernel_cols) weights and global gradient.
    float *l1_weights = (float*)malloc(l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float));
    // Random initialize weights
    random_init(l1_weights, l1_filters * l1_kernel_rows * l1_kernel_cols, 0, 1);
    
    // Layer 1 device
    float *dev_l1_vals, *dev_l1_outs, *dev_l1_pdL_pdouts, *dev_l1_douts_dvals, *dev_l1_pdL_pdvals,
          *dev_l1_pdL_pdouts_prev, *dev_l1_weights, *dev_l1_grads;
    
    if (cudaMalloc(&dev_l1_vals, l1_filters * l1_out_rows * l1_out_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_outs, l1_filters * l1_out_rows * l1_out_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_pdL_pdouts, l1_filters * l1_out_rows * l1_out_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_douts_dvals, l1_filters * l1_out_rows * l1_out_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_pdL_pdvals, l1_filters * l1_out_rows * l1_out_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_pdL_pdouts_prev, l1_in_rows * l1_in_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_weights, l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l1_grads, l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float)) != cudaSuccess) {
        printf("Could not allocate layer 1 memory on device.\n");
        return EXIT_FAILURE;
    }

    /** Layer2: Dense Layer **/

    // CONFIGURABLE
    int l2_out_nodes = 128;
    int l2_activation = 0; // the code for sigmoid

    // Input nodes depend on the previous layer
    int l2_in_nodes = l1_filters * l1_out_rows * l1_out_cols;
    
    // Each out-node has a weight for each in-node
    float *l2_weights = (float*)malloc(l2_in_nodes * l2_out_nodes * sizeof(float));
    // Random initialize weights
    random_init(l2_weights, l2_in_nodes * l2_out_nodes, -1, 1);

    float *dev_l2_vals, *dev_l2_outs, *dev_l2_pdL_pdouts, *dev_l2_douts_dvals, *dev_l2_pdL_pdvals,
          *dev_l2_weights, *dev_l2_grads;

    if (cudaMalloc(&dev_l2_vals, l2_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l2_outs, l2_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l2_pdL_pdouts, l2_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l2_douts_dvals, l2_out_nodes* sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l2_pdL_pdvals, l2_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l2_weights, l2_in_nodes * l2_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l2_grads, l2_in_nodes * l2_out_nodes * sizeof(float)) != cudaSuccess) {
        printf("Could not allocate layer 2 memory on device.\n");
        return EXIT_FAILURE;
    }

    /** Layer3: Dense Layer (Output) **/

    int l3_out_nodes = 15; // 15 classes to predict
    int l3_activation = 2; // the code for none; no activation
                           // the cross entropy is programmed to apply softmax activation

    // Input nodes depend on the previous layer
    int l3_in_nodes = l2_out_nodes;
    
    // The nodes need pre-activation vals and activated outs.
    float *l3_vals =        (float*)malloc(l3_out_nodes * sizeof(float));
    float *l3_outs =        (float*)malloc(l3_out_nodes * sizeof(float));
    float *l3_pdL_pdvals =  (float*)malloc(l3_out_nodes * sizeof(float));
    
    // Each out-node has a weight for each in-node
    float *l3_weights = (float*)malloc(l3_in_nodes * l3_out_nodes * sizeof(float));
    // Random initialize weights
    random_init(l3_weights, l3_in_nodes * l3_out_nodes, -1, 1);

    float *dev_l3_vals, *dev_l3_outs, *dev_l3_pdL_pdouts, *dev_l3_douts_dvals, *dev_l3_pdL_pdvals,
          *dev_l3_weights, *dev_l3_grads;

    if (cudaMalloc(&dev_l3_vals, l3_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l3_outs, l3_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l3_pdL_pdouts, l3_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l3_douts_dvals, l3_out_nodes* sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l3_pdL_pdvals, l3_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l3_weights, l3_in_nodes * l3_out_nodes * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_l3_grads, l3_in_nodes * l3_out_nodes * sizeof(float)) != cudaSuccess) {
        printf("Could not allocate layer 3 memory on device.\n");
        return EXIT_FAILURE;
    }

    // Copy all data to device
    float *dev_data;
    if (cudaMalloc(&dev_data, 64 * 64 * 14999 * sizeof(float)) != cudaSuccess) {
        printf("Error allocating memory for images on device.\n");
        return EXIT_FAILURE;
    }
    if (cudaMemcpy(dev_data, data, 64 * 64 * 14999 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Error transferring images to device.\n");
        return EXIT_FAILURE;
    }


    // Memory for training targets
    float target_vec[15];
    // Memory for losses
    float current_loss;
    int summary_iterations = 500;
    float past_losses[summary_iterations];
    bool past_correct[summary_iterations];


    // FILE PATHS
    const char *l1_weights_filename = "data/model/l1_weights.data";
    const char *l2_weights_filename = "data/model/l2_weights.data";
    const char *l3_weights_filename = "data/model/l3_weights.data";

    if (load) {
        printf("Loading from existing model.\n");
        FILE *l1_weights_file = fopen(l1_weights_filename, "rb");
        FILE *l2_weights_file = fopen(l2_weights_filename, "rb");
        FILE *l3_weights_file = fopen(l3_weights_filename, "rb");
        if (l1_weights_file == NULL) {
            printf("Layer 1 weights not found. Using randomly initialized.\n");
        } else {
            fread(l1_weights, sizeof(float), l1_filters * l1_kernel_rows * l1_kernel_cols, l1_weights_file);
            fclose(l1_weights_file);
        }
        if (l2_weights_file == NULL) {
            printf("Layer 2 weights not found. Using randomly initialized.\n");
        } else {
            fread(l2_weights, sizeof(float), l2_in_nodes * l2_out_nodes, l2_weights_file);
            fclose(l2_weights_file);
        }
        if (l3_weights_file == NULL) {
            printf("Layer 3 weights not found. Using randomly initialized.\n");
        } else {
            fread(l3_weights, sizeof(float), l3_in_nodes * l3_out_nodes, l3_weights_file);
            fclose(l3_weights_file);
        }
    }

    // Copy all weights to device
    if (cudaMemcpy(dev_l1_weights, l1_weights,
            l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Layer 1 weights failed to copy to device.\n");
        return EXIT_FAILURE;
    }
    if (cudaMemcpy(dev_l2_weights, l2_weights,
            l2_in_nodes * l2_out_nodes * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Layer 2 weights failed to copy to device.\n");
        return EXIT_FAILURE;
    }
    if (cudaMemcpy(dev_l3_weights, l3_weights,
            l3_in_nodes * l3_out_nodes * sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("Layer 3 weights failed to copy to device.\n");
        return EXIT_FAILURE;
    }

    struct timespec begin_time, end_time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &begin_time);

    int total_iters = atoi(iters);
    for (int iter = 0; iter < total_iters; ++iter) {
        int sample_index = rand() % 14999; // 14999 images, indices 0 - 14998
        int float_offset = sample_index * l1_in_rows * l1_in_cols;
        int target_label = metadata[sample_index].label;
        
        // Layer 1 forward
        Conv2D_forward(
            dev_l1_outs, l1_out_rows, l1_out_cols,
            dev_l1_vals,
            dev_l1_douts_dvals,
            dev_data + float_offset, l1_in_rows, l1_in_cols,
            dev_l1_weights, l1_kernel_rows, l1_kernel_cols,
            l1_stride_rows, l1_stride_cols, l1_filters,
            l1_activation
        );
        // Layer 2 forward
        Dense_forward(
            dev_l2_outs, l2_out_nodes,
            dev_l2_vals,
            dev_l2_douts_dvals,
            dev_l1_outs, l2_in_nodes,
            dev_l2_weights,
            l2_activation
        );
        // Layer 3 forward
        Dense_forward(
            dev_l3_outs, l3_out_nodes,
            dev_l3_vals,
            dev_l3_douts_dvals,
            dev_l2_outs, l3_in_nodes,
            dev_l3_weights,
            l3_activation
        );

        /* Calculate loss */

        cn_mnist_target_to_vec(target_vec, target_label); // create target one-hot vector

        // Move needed data back to host
        if (cudaMemcpy(l3_vals, dev_l3_vals, l3_out_nodes * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error copying l3_vals back to host.\n");
            return EXIT_FAILURE;
        }

        /* This doesn't need to be copied because it's just a working buffer for cat_cross_entropy()
        if (cudaMemcpy(l3_outs, dev_l3_outs, l3_out_nodes * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error copying l3_outs back to host.\n");
            return EXIT_FAILURE;
        }*/
        
        // categorical cross entropy loss with softmax activation
        cat_cross_entropy(
            l3_out_nodes, target_vec, l3_vals, l3_outs,
            l3_pdL_pdvals, &current_loss
        );
        
        // Move newly calculated gradient to device
        if (cudaMemcpy(dev_l3_pdL_pdvals, l3_pdL_pdvals, l3_out_nodes * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("Error copying l3_pdL_pdvals to device.\n");
            return EXIT_FAILURE;
        }

        // Store aggregate measures
        int predicted_index = argmax(l3_outs, 15);
        past_correct[(iter % summary_iterations)] = (target_vec[predicted_index] == 1);
        past_losses[(iter % summary_iterations)] = current_loss;

        // Layer 3 backward
        Dense_backward(
            l3_out_nodes,
            dev_l3_pdL_pdouts,
            dev_l3_douts_dvals,
            dev_l3_pdL_pdvals,
            dev_l2_pdL_pdouts,
            dev_l2_outs, l3_in_nodes,
            dev_l3_weights,
            dev_l3_grads,
            l3_activation
        );
        // Layer 2 backward
        Dense_backward(
            l2_out_nodes,
            dev_l2_pdL_pdouts,
            dev_l2_douts_dvals,
            dev_l2_pdL_pdvals,
            dev_l1_pdL_pdouts,
            dev_l1_outs, l2_in_nodes,
            dev_l2_weights,
            dev_l2_grads,
            l2_activation
        );
        // Layer 1 backward
        Conv2D_backward(
            l1_out_rows, l1_out_cols,
            dev_l1_pdL_pdouts,
            dev_l1_douts_dvals,
            dev_l1_pdL_pdvals,
            dev_l1_pdL_pdouts_prev,
            dev_data + float_offset, l1_in_rows, l1_in_cols,
            dev_l1_weights, l1_kernel_rows, l1_kernel_cols,
            dev_l1_grads,
            l1_stride_rows, l1_stride_cols, l1_filters
        );

        // Lower learning rate (alpha) over time.
        // Kind of like simulated annealing.
        float alpha = 0.1 - (0.05 * (float)iter / (float)total_iters);

        // Update layer 1
        SGD_update_params(
            alpha, dev_l1_weights, dev_l1_grads,
            l1_filters * l1_kernel_rows * l1_kernel_cols
        );
        // Update layer 2
        SGD_update_params(
            alpha, dev_l2_weights, dev_l2_grads,
            l2_in_nodes * l2_out_nodes
        );
        // Update layer 3
        SGD_update_params(
            alpha, dev_l3_weights, dev_l3_grads,
            l3_in_nodes * l3_out_nodes
        );

        // This will execute once every summary_iterations.
        if ( (iter+1) % summary_iterations == 0) {
            clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
            uint64_t time_over_period = // in microseconds
                (end_time.tv_sec - begin_time.tv_sec) * 1000000
                + (end_time.tv_nsec - begin_time.tv_nsec) / 1000;


            // important sanity check: make sure offsets are correct.
            // images and metadata should make sense and match.
            flt_img_to_ascii(data+float_offset, 64, 64, 1);
            printf(
                "Image file: %s. Correct label:  %d\n\n",
                metadata[sample_index].input,
                metadata[sample_index].label
            );

            // Pretty print the predictions table
            print_predictions(l3_outs);
            
            float avg_loss = 0;
            float accuracy = 0;
            for (int i = 0; i < summary_iterations; ++i) {
                avg_loss += past_losses[i];
                accuracy += (past_correct[i] ? 1 : 0);
            }
            avg_loss /= summary_iterations;
            accuracy = accuracy * 100 /  summary_iterations;
            printf("\nCurrent iteration: %d\n", iter + 1);
            printf("Average loss over last %d iterations: %f\n", summary_iterations, avg_loss);
            printf("Accuracy of last %d iterations: %0.2f%%\n", summary_iterations, accuracy);
            printf("Time spent processing the last %d iterations: %0.2fms\n",
                summary_iterations, time_over_period / 1000.0);

            // Mark the time for the next period
            clock_gettime(CLOCK_MONOTONIC_RAW, &begin_time);
        }
    }
    
    if (save) {
        // Copy all weights back to host
        if (cudaMemcpy(l1_weights, dev_l1_weights,
            l1_filters * l1_kernel_rows * l1_kernel_cols * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Layer 1 weights failed to copy back to host.\n");
            return EXIT_FAILURE;
        }
        if (cudaMemcpy(l2_weights, dev_l2_weights,  
            l2_in_nodes * l2_out_nodes * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Layer 2 weights failed to copy back to host.\n");
            return EXIT_FAILURE;
        }
        if (cudaMemcpy(l3_weights, dev_l3_weights,
            l3_in_nodes * l3_out_nodes * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Layer 3 weights failed to copy back to host.\n");
            return EXIT_FAILURE;
        }

        printf("Saving model.\n");
        FILE *l1_weights_file = fopen(l1_weights_filename, "wb");
        FILE *l2_weights_file = fopen(l2_weights_filename, "wb");
        FILE *l3_weights_file = fopen(l3_weights_filename, "wb");
        fwrite(l1_weights, sizeof(float), l1_filters * l1_kernel_rows * l1_kernel_cols, l1_weights_file);
        fwrite(l2_weights, sizeof(float), l2_in_nodes * l2_out_nodes, l2_weights_file);
        fwrite(l3_weights, sizeof(float), l3_in_nodes * l3_out_nodes, l3_weights_file);
        fclose(l1_weights_file);
        fclose(l2_weights_file);
        fclose(l3_weights_file);
    }

    // Free host memory
    free(l1_weights); free(l2_weights); free(l3_weights);
    free(l3_vals); free(l3_outs); free(l3_pdL_pdvals);

    // TODO: free device memory properly
    cudaFree(dev_data);
    cudaDeviceReset();

    return 0;
}

void interactive_loop() {
    
    InputLabel *metadata = NULL;
    float *chinese_mnist_processed = NULL;
    char buf[1024];
    while (true) {
        print_prompt();
        fgets(buf, 1023, stdin);
            
        Tokens *toks = Tokens_create();
        
        // If tokens not fetched successfully
        if (!Tokens_fetch(toks, buf) || buf[0] == '\n') {
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
        
        else if (cmd == PREDICT) {
            cmd_predict(toks);
        }

        else if (cmd == PREPROCESS) {
            cmd_preprocess(&metadata, &chinese_mnist_processed);
        }

        else if (cmd == SHOW) {
            cmd_show(toks, chinese_mnist_processed);
        }

        else if (cmd == TRAIN) {
            int status = cmd_train(metadata, chinese_mnist_processed, toks);
            if (status == EXIT_FAILURE)
                return;
        }

        toks = Tokens_destroy(toks);
    }
}

int main(int argc, char *argv[]) {
    srand( 479 ); // seed PRNG for reproducible results
    interactive_loop();
    return 0;
}
