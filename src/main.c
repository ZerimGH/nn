#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

#define BATCH_SIZE 100

static inline double randf(double min, double max) {
    double r = (double)rand();
    r /= (double)RAND_MAX;
    r *= (max - min);
    r += min;
    return r;
}

static inline double xorf(double a, double b) {
    int ia = a >= 0.5;
    int ib = b >= 0.5;

    return (double)!((ia && ib) || (!ia && !ib));
}

int main(void) {
    srand(time(NULL));
    size_t layer_count = 3;
    size_t layer_sizes[] = {2, 4, 1};
    nn_network_t *net = nn_network_create(layer_count, layer_sizes, NN_SIGMOID);
    // nn_network_t *net = nn_network_load("xor.nn");
    if (!net) {
        fprintf(stderr, "Failed to create neural network\n");
        return 1;
    }

    double inputs[BATCH_SIZE][2];
    double targets[BATCH_SIZE][1];
    double *input_ptrs[BATCH_SIZE];
    double *target_ptrs[BATCH_SIZE];

    for (size_t i = 0; i < BATCH_SIZE; i++) {
        inputs[i][0] = randf(0.0, 1.0);
        inputs[i][1] = randf(0.0, 1.0);
        targets[i][0] = xorf(inputs[i][0], inputs[i][1]);
        input_ptrs[i] = inputs[i];
        target_ptrs[i] = targets[i];
    }

    printf("Training...\n");
    size_t iterations = 200000;
    double learning_rate = 0.125;
    if (nn_network_train(net, BATCH_SIZE, input_ptrs, 2, target_ptrs, 1, iterations, learning_rate)) {
        fprintf(stderr, "Failed to train neural network\n");
        nn_network_destroy(net);
        return 1;
    }

    printf("Training finished, saving network\n");
    if (nn_network_save(net, "xor.nn")) {
        fprintf(stderr, "Failed to save network\n");
        return 1;
    }

    printf("Saving finished, testing\n");
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        double outputs[1] = {0.f};
        if (nn_network_evaluate(net, 2, inputs[i], 1, outputs)) {
            fprintf(stderr, "Failed to evaluate neural network\n");
            nn_network_destroy(net);
            return 1;
        }
        printf("(Inputs: (%f, %f), Expected: %f) -> (Output: %f, Error: %f)\n", inputs[i][0], inputs[i][1],
               targets[i][0], outputs[0], targets[i][0] - outputs[0]);
    }

    nn_network_destroy(net);
    net = NULL;
    return 0;
}
