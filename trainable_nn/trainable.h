/**
 * @file trainable.h
 * @author Michael Gathara (michael@michaelgathara.com)
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once
#include <vector>
#include <random>

class NeuralNetwork {
private:
    std::vector<std::vector<double>> hidden_weights;
    std::vector<double> output_weights;
    double hidden_bias;
    double output_bias;
    double learning_rate;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

public:
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes, double learningr);


    static double sigmoid(double x);

    std::vector<double> forward(const std::vector<double>& input);

    /**
     * ADDING TRAINING BELOW
     */
    double compute_loss(double predicted, double actual);

    void update_weights(const std::vector<double>& input, double target);

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets, int epochs);

};