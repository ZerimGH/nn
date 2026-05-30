#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double randf(double min, double max) {
    double r = (double)rand();
    r /= (double)RAND_MAX;
    r *= (max - min);
    r += min;
    return r;
}

double nn_relu(double x) {
    return x > 0 ? x : 0.0;
}

double nn_drelu(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double nn_leaky_relu(double x) {
    return x > 0 ? x : x * 0.01;
}

double nn_dleaky_relu(double x) {
    return x > 0 ? 1.0 : 0.01;
}

double nn_sigmoid(double x) {
    return 1.0 / (1.0 + expf(-x));
}

double nn_dsigmoid(double x) {
    return x * (1.0 - x);
}

nn_network_t *nn_network_create(size_t layer_count, size_t *layer_sizes, nn_func_type_t func_type) {
    if (layer_count == 0 || !layer_sizes) return NULL;

    /* Allocate space for the network struct */
    nn_network_t *net = malloc(sizeof(nn_network_t));
    if (!net) return NULL;

    /* Set activation function pointers */
    switch (func_type) {
        case NN_SIGMOID:
            net->func = nn_sigmoid;
            net->dfunc = nn_dsigmoid;
            break;
        case NN_RELU:
            net->func = nn_relu;
            net->dfunc = nn_drelu;
            break;
        case NN_LEAKY_RELU:
            net->func = nn_leaky_relu;
            net->dfunc = nn_dleaky_relu;
            break;
        default: free(net); return NULL;
    }
    net->func_type = func_type;

    /* Create all layers with random initial weights and biases */
    net->layers = calloc(layer_count - 1, sizeof(nn_layer_t));
    if (!net->layers) {
        free(net);
        return NULL;
    }
    net->layer_count = layer_count - 1;
    net->max_out_count = 0;

    size_t in_count = layer_sizes[0];
    for (size_t i = 1; i < layer_count; i++) {
        nn_layer_t *layer = &net->layers[i - 1];
        size_t out_count = layer_sizes[i];
        size_t weight_count = in_count * out_count;
        size_t bias_count = out_count;
        size_t activation_count = out_count;

        /* Update max_out_count */
        if (out_count > net->max_out_count) net->max_out_count = out_count;

        /* Allocate arrays */
        layer->weights = malloc(sizeof(double) * weight_count);
        if (!layer->weights) goto layer_failure;
        layer->biases = malloc(sizeof(double) * bias_count);
        if (!layer->biases) goto layer_failure;
        layer->activations = malloc(sizeof(double) * activation_count);
        if (!layer->activations) goto layer_failure;
        layer->in_count = in_count;
        layer->out_count = out_count;

        /* Initialise weights, biases and activations */
        double limit = 0.0;

        switch (net->func_type) {
            case NN_RELU:
            case NN_LEAKY_RELU: limit = sqrt(6.0 / (double)in_count); break;

            case NN_SIGMOID:
            default: limit = sqrt(6.0 / (double)(in_count + out_count)); break;
        }

        for (size_t j = 0; j < weight_count; j++) {
            layer->weights[j] = randf(-limit, limit);
        }

        for (size_t j = 0; j < bias_count; j++) {
            layer->biases[j] = 0.0;
        }

        for (size_t j = 0; j < activation_count; j++) {
            layer->activations[j] = 0.0;
        }

        /* Go to next layer */
        in_count = out_count;
        continue;

    layer_failure:
        /* Free current layer */
        if (layer->weights) free(layer->weights);
        if (layer->biases) free(layer->biases);
        if (layer->activations) free(layer->activations);
        /* Free all previous layers */
        for (size_t j = 1; j < i; j++) {
            nn_layer_t *p_layer = &net->layers[j - 1];
            if (p_layer->weights) free(p_layer->weights);
            if (p_layer->biases) free(p_layer->biases);
            if (p_layer->activations) free(p_layer->activations);
        }
        /* Free rest of network and return NULL */
        free(net->layers);
        free(net);
        return NULL;
    }

    return net;
}

void nn_network_destroy(nn_network_t *net) {
    if (!net) return;
    if (net->layers) {
        for (size_t i = 0; i < net->layer_count; i++) {
            nn_layer_t *layer = &net->layers[i];
            if (layer->weights) free(layer->weights);
            if (layer->biases) free(layer->biases);
            if (layer->activations) free(layer->activations);
        }
        free(net->layers);
        net->layers = NULL;
    }
    free(net);
}

int nn_network_evaluate(nn_network_t *net, size_t input_count, double *inputs, size_t output_count, double *outputs) {
    if (!net || !inputs || !net->func) return 1;
    /* Missing outputs is not an error, outputs will be unused internally
     * for training */
    if (input_count != net->layers[0].in_count) return 1;

    if (outputs && (output_count != net->layers[net->layer_count - 1].out_count)) return 1;

    size_t layer_input_count = input_count;
    double *layer_inputs = inputs;
    for (size_t i = 0; i < net->layer_count; i++) {
        nn_layer_t *layer = &net->layers[i];
        for (size_t j = 0; j < layer->out_count; j++) {
            /* Calculate weighted sum of inputs */
            double weighted_sum = 0;
            for (size_t k = 0; k < layer_input_count; k++) {
                double weight = layer->weights[k * layer->out_count + j];
                weighted_sum += layer_inputs[k] * weight;
            }
            /* Add the bias */
            weighted_sum += layer->biases[j];
            /* Activation function */
            weighted_sum = net->func(weighted_sum);
            layer->activations[j] = weighted_sum;
        }
        layer_input_count = layer->out_count;
        layer_inputs = layer->activations;
    }

    if (outputs) memcpy(outputs, net->layers[net->layer_count - 1].activations, sizeof(double) * output_count);

    return 0;
}

static int nn_network_backprop(nn_network_t *net, size_t input_count, double *inputs, size_t target_count,
                               double *targets, double learning_rate, double *deltas, double *next_deltas) {
    (void)target_count;
    /* Evaluate network on inputs to calculate error */
    if (nn_network_evaluate(net, input_count, inputs, 0, NULL)) return 1;

    /* Go backwards through layers adjusting weights and biases */
    for (size_t i = net->layer_count; i > 0; i--) {
        size_t idx = i - 1;
        nn_layer_t *layer = &net->layers[idx];
        double *layer_inputs = (idx == 0) ? inputs : net->layers[idx - 1].activations;

        if (idx == net->layer_count - 1) {
            /* Deltas for outputs are just error * dfunc(activation) */
            for (size_t j = 0; j < layer->out_count; j++) {
                double error = layer->activations[j] - targets[j];
                deltas[j] = error * net->dfunc(layer->activations[j]);
            }
        } else {
            /* Deltas for hidden layers propagate errors from next layer up */
            nn_layer_t *layer_next = &net->layers[idx + 1];
            for (size_t j = 0; j < layer->out_count; j++) {
                double error = 0.f;
                for (size_t k = 0; k < layer_next->out_count; k++) {
                    error += next_deltas[k] * layer_next->weights[j * layer_next->out_count + k];
                }
                deltas[j] = error * net->dfunc(layer->activations[j]);
            }
        }

        /* Update weights and biases based on deltas */
        for (size_t j = 0; j < layer->out_count; j++) {
            layer->biases[j] -= deltas[j] * learning_rate;
            for (size_t k = 0; k < layer->in_count; k++) {
                double grad = deltas[j] * layer_inputs[k];
                layer->weights[k * layer->out_count + j] -= grad * learning_rate;
            }
        }

        /* Update deltas for the next layer up */
        for (size_t j = 0; j < layer->out_count; j++) {
            next_deltas[j] = deltas[j];
        }
    }
    return 0;
}

int nn_network_train(nn_network_t *net, size_t batch_size, double **batch_inputs, size_t inputs_size,
                     double **batch_targets, size_t targets_size, size_t iterations, double learning_rate) {
    if (!net || !batch_inputs || !batch_targets) return 1;
    if (batch_size == 0) return 1;
    if (inputs_size != net->layers[0].in_count) return 1;
    if (targets_size != net->layers[net->layer_count - 1].out_count) return 1;

    /* Allocate delta buffers */
    double *deltas = calloc(net->max_out_count, sizeof(double));
    if (!deltas) return 1;

    double *next_deltas = calloc(net->max_out_count, sizeof(double));
    if (!next_deltas) {
        free(deltas);
        return 1;
    }

    /* Run backpropagation on each sample iteration times */
    /* TODO: Maybe like randomise the order of samples or randomly select? */
    for (size_t iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < batch_size; i++) {
#if 0
            size_t idx = i;
#else
            size_t idx = rand() % batch_size;
#endif
            if (nn_network_backprop(net, inputs_size, batch_inputs[idx], targets_size, batch_targets[idx],
                                    learning_rate, deltas, next_deltas)) {
                free(deltas);
                free(next_deltas);
                return 1;
            }
        }
    }

    free(deltas);
    free(next_deltas);
    return 0;
}

int nn_network_save(nn_network_t *net, const char *path) {
    if (!net || !path) return 1;
    FILE *file = fopen(path, "wb");
    if (!file) goto err;
    /* Write some kinda unique header */
    if (fprintf(file, "nn_network_t") < 0) goto err;

    /* Write information of the nn_network_t struct */
    if (fwrite((char *)&net->layer_count, 1, sizeof(net->layer_count), file) != sizeof(net->layer_count)) goto err;
    if (fwrite((char *)&net->max_out_count, 1, sizeof(net->max_out_count), file) != sizeof(net->max_out_count))
        goto err;
    if (fwrite((char *)&net->func_type, 1, sizeof(net->func_type), file) != sizeof(net->func_type)) goto err;

    /* Write layer information */
    for (size_t i = 0; i < net->layer_count; i++) {
        nn_layer_t *layer = &net->layers[i];
        if (fwrite((char *)&layer->in_count, 1, sizeof(layer->in_count), file) != sizeof(layer->in_count)) goto err;
        if (fwrite((char *)&layer->out_count, 1, sizeof(layer->out_count), file) != sizeof(layer->out_count)) goto err;

        /* Write biases */
        if (fwrite(layer->biases, sizeof(double), layer->out_count, file) != layer->out_count) goto err;

        /* Write weights */
        size_t total_weights = layer->in_count * layer->out_count;
        if (fwrite(layer->weights, sizeof(double), total_weights, file) != total_weights) goto err;
    }

    fclose(file);
    return 0;

err:
    if (file) fclose(file);
    return 1;
}

nn_network_t *nn_network_load(const char *path) {
    if (!path) return NULL;
    FILE *file = fopen(path, "rb");
    nn_network_t *net = calloc(1, sizeof(nn_network_t));
    if (!file || !net) goto err;

    /* Check header */
    char header_buf[12] = {0};
    if (fread(header_buf, sizeof(char), 12, file) != 12) goto err;
    if (memcmp(header_buf, "nn_network_t", 12)) goto err;

    /* Read information of the nn_network_t struct */
    if (fread((char *)&net->layer_count, 1, sizeof(net->layer_count), file) != sizeof(net->layer_count)) goto err;
    if (fread((char *)&net->max_out_count, 1, sizeof(net->max_out_count), file) != sizeof(net->max_out_count)) goto err;
    if (fread((char *)&net->func_type, 1, sizeof(net->func_type), file) != sizeof(net->func_type)) goto err;

    /* Set activation function pointers */
    switch (net->func_type) {
        case NN_SIGMOID:
            net->func = nn_sigmoid;
            net->dfunc = nn_dsigmoid;
            break;
        case NN_RELU:
            net->func = nn_relu;
            net->dfunc = nn_drelu;
            break;
        case NN_LEAKY_RELU:
            net->func = nn_leaky_relu;
            net->dfunc = nn_dleaky_relu;
            break;
        default: goto err;
    }

    /* Read layer information */
    net->layers = calloc(net->layer_count, sizeof(nn_layer_t));
    if (!net->layers) goto err;

    for (size_t i = 0; i < net->layer_count; i++) {
        nn_layer_t *layer = &net->layers[i];
        if (!layer) goto err;
        if (fread((char *)&layer->in_count, 1, sizeof(layer->in_count), file) != sizeof(layer->in_count)) goto err;
        if (fread((char *)&layer->out_count, 1, sizeof(layer->out_count), file) != sizeof(layer->out_count)) goto err;

        /* Read biases */
        layer->biases = malloc(sizeof(double) * layer->out_count);
        if (!layer->biases) goto err;
        if (fread(layer->biases, sizeof(double), layer->out_count, file) != layer->out_count) {
            free(layer->biases);
            goto err;
        }

        /* Read weights */
        size_t total_weights = layer->in_count * layer->out_count;
        layer->weights = malloc(sizeof(double) * total_weights);
        if (!layer->weights) goto err;
        if (fread(layer->weights, sizeof(double), total_weights, file) != total_weights) {
            free(layer->biases);
            free(layer->weights);
            goto err;
        }

        /* Initialise weights, biases and activations */
        layer->activations = calloc(layer->out_count, sizeof(double));
        if (!layer->activations) {
            free(layer->biases);
            free(layer->weights);
            goto err;
        }
    }

    size_t ptr = ftell(file);
    fseek(file, 0, SEEK_END);
    size_t end_ptr = ftell(file);
    if (ptr != end_ptr) goto err; /* Full file was not read */

    fclose(file);
    return net;

err:
    if (file) fclose(file);
    if (net) {
        for (size_t i = 0; i < net->layer_count; i++) {
            if (net->layers[i].biases) free(net->layers[i].biases);
            if (net->layers[i].weights) free(net->layers[i].weights);
            if (net->layers[i].activations) free(net->layers[i].activations);
        }
        if (net->layers) free(net->layers);
        free(net);
    }
    return NULL;
}
