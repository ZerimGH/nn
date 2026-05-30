#ifndef NN_H
#define NN_H

#include <stddef.h>

/* --- TYPES --- */
typedef enum { NN_SIGMOID, NN_RELU, NN_LEAKY_RELU } nn_func_type_t;

typedef double (*nn_activation_function)(double);

typedef struct {
    size_t in_count, out_count;
    double *weights, *biases; /* Weights is a flattened 2d array, 
                                 indexed by (in * out_count + out) */
    double *activations;
} nn_layer_t;

typedef struct {
    size_t layer_count;
    nn_layer_t *layers;
    nn_func_type_t func_type;
    nn_activation_function func;
    nn_activation_function dfunc;

    size_t max_out_count; /* Store the maximum number of outputs of all
                             the layers to help with allocation */
} nn_network_t;

/* --- NETWORKS --- */
/* Create a neural network
 * layer_sizes is an array of size layer_count of the sizes of each layer 
 * (including inputs and outputs)
 * func_type is the type of activation function that will be used for training */
nn_network_t *nn_network_create(size_t layer_count, size_t *layer_sizes, nn_func_type_t func_type);
/* Destroy a neural network, frees all memory allocated for the network */
void nn_network_destroy(nn_network_t *net);
/* Evaluate a neural network on a set of inputs 
 * inputs is an array of size input_count of the input data to be used
 * outputs is an array of size output_count that the result will be written to */
int nn_network_evaluate(nn_network_t *net, size_t input_count, double *inputs, size_t output_count, double *outputs);
/* Train a neural network on a batch of data 
 * batch_inputs and batch_targets are arrays of size batch_size containing pointers to the arrays used as inputs and expected outputs
 * iterations is how many iterations the network will train for
 * learning_rate is a value to control how quickly the network learns, bigger
 * values cause bigger changes to the network but it may cause inaccuracy */
int nn_network_train(nn_network_t *net, size_t batch_size, double **batch_inputs, size_t inputs_size,
                     double **batch_targets, size_t targets_size, size_t iterations, double learning_rate);
/* Save a network to a file */
int nn_network_save(nn_network_t *net, const char *path);
/* Load a network from a file */
nn_network_t *nn_network_load(const char *path);

#endif /* NN_H */
