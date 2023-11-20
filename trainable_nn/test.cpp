#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "trainable.h"

std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line, cell;
    bool header_row = true;

    while (std::getline(file, line)) {
        if (header_row) {
            header_row = false;
            continue;
        }
        std::vector<double> row;
        std::stringstream lineStream(line);
        while (std::getline(lineStream, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << cell << std::endl;
                continue;
            }
        }
        data.push_back(row);
    }
    return data;
}


int main() {
    
    std::string training_file = "-8976732775143038538.csv";
    std::string testing_file = "testing_data.csv";

    std::vector<std::vector<double>> training_data = read_csv(training_file);
    std::vector<std::vector<double>> testing_data = read_csv(testing_file);

    std::vector<std::vector<double>> training_inputs(training_data.size());
    std::vector<double> training_targets(training_data.size());

    for (int i = 0; i < training_data.size(); i++) {
        training_inputs[i] = std::vector<double>(training_data[i].begin(), training_data[i].end() - 1);
        training_targets[i] = training_data[i].back();
    }

    std::vector<std::vector<double>> testing_inputs(testing_data.size());
    std::vector<double> testing_targets(testing_data.size());

    for (int i = 0; i < testing_data.size(); i++) {
        testing_inputs[i] = std::vector<double>(testing_data[i].begin(), testing_data[i].end() - 1);
        testing_targets[i] = testing_data[i].back();
    }

    NeuralNetwork nn(5, 10, 1, 0.01);
    nn.train(training_inputs, training_targets, 1000);

    double total_loss = 0.0;
    for (int i = 0; i < testing_inputs.size(); i++) {
        double predicted = nn.forward(testing_inputs[i])[0];
        double actual = testing_targets[i];
        double loss = nn.compute_loss(predicted, actual);
        total_loss += loss;
    }
    std::cout << "Average loss: " << total_loss / testing_inputs.size() << std::endl;

    return 0;
}