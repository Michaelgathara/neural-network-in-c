/**
 * @file neural.cpp
 * @author Michael Gathara (michael@michaelgathara.com)
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "trainable.h"
#include <iostream>
#include <cmath>

NeuralNetwork::NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, double learningr) : learning_rate(learningr), gen(std::random_device{}()), dis(-1.0, 1.0) {
    hidden_weights.resize(hidden_nodes, std::vector<double>(input_nodes));
    for (int i = 0; i < hidden_nodes; i++) {
        for (int j = 0; j < input_nodes; j++) {
            hidden_weights[i][j] = dis(gen);
        }
    }

    output_weights.resize(output_nodes);
    for (int i = 0; i < output_nodes; i++) {
        output_weights[i] = dis(gen);
    }

    hidden_bias = dis(gen);
    output_bias = dis(gen);
}


double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> hidden_values(hidden_weights.size(), 0.0);
    for (int i = 0; i < hidden_weights.size(); i++) {
        for (int j = 0; j < input.size(); j++) {
            hidden_values[i] += input[j] * hidden_weights[i][j];
        }
        hidden_values[i] += hidden_bias;
        hidden_values[i] = sigmoid(hidden_values[i]);
    }

    double output_value = 0.0;
    for (int i = 0; i < hidden_values.size(); i++) {
        output_value += hidden_values[i] * output_weights[i];
    }
    output_value += output_bias;
    output_value = sigmoid(output_value);

    return {output_value};
}

/**
 * ADDING TRAINING BELOW
 */
double NeuralNetwork::compute_loss(double predicted, double actual) {
    return 0.5 * pow((predicted - actual), 2);
}

void NeuralNetwork::update_weights(const std::vector<double>& input, double target) {
    std::vector<double> hidden_output = forward(input);

    double predicted = hidden_output[0];
    double error = predicted - target;

    double d_predicted = error * predicted * (1 - predicted);

    for (int i = 0; i < output_weights.size(); i++) {
        output_weights[i] -= learning_rate * d_predicted * hidden_output[i];
    }
    output_bias -= learning_rate * d_predicted;

    for (int i = 0; i < hidden_weights.size(); i++) {
        double d_hidden = d_predicted * output_weights[i] * hidden_output[i] * (1 - hidden_output[i]);
        for (int j = 0; j < input.size(); j++) {
            hidden_weights[i][j] -= learning_rate * d_hidden * input[j];
        }
        hidden_bias -= learning_rate * d_hidden;
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            std::vector<double> output = forward(inputs[i]);
            total_loss += compute_loss(output[0], targets[i]);
            update_weights(inputs[i], targets[i]);
        }
        std::cout.precision(std::numeric_limits<double>::max_digits10 - 1);
        std::cout << "Epoch " << epoch << " Loss: " << std::scientific << total_loss / inputs.size() << '\n' << std::flush;
    }
}