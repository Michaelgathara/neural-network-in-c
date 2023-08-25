#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class NeuralNetwork {
private:
    std::vector<std::vector<double>> weights_hidden;
    std::vector<double> weights_output;
    double hidden_bias;
    double output_bias;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

public:
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) : gen(std::random_device{}()), dis(-1.0, 1.0) {
        weights_hidden.resize(hidden_nodes, std::vector<double>(input_nodes));
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
                weights_hidden[i][j] = dis(gen);
            }
        }

        weights_output.resize(output_nodes);
        for (int i = 0; i < output_nodes; i++) {
            weights_output[i] = dis(gen);
        }

        hidden_bias = dis(gen);
        output_bias = dis(gen);
    }


    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> hidden_values(weights_hidden.size(), 0.0);
        for (int i = 0; i < weights_hidden.size(); i++) {
            for (int j = 0; j < input.size(); j++) {
                hidden_values[i] += input[j] * weights_hidden[i][j];
            }
            hidden_values[i] += hidden_bias;
            hidden_values[i] = sigmoid(hidden_values[i]);
        }

        double output_value = 0.0;
        for (int i = 0; i < hidden_values.size(); i++) {
            output_value += hidden_values[i] * weights_output[i];
        }
        output_value += output_bias;
        output_value = sigmoid(output_value);

        return {output_value};
    }
};

int main() {
//    NeuralNetwork nn(3, 4, 1);
    NeuralNetwork nn(5, 10, 1);
    std::vector<double> input = {0.5, 0.3, 0.2, 0.2, 0.1};
    std::vector<double> output = nn.forward(input);

    std::cout << "Output: " << output[0] << std::endl;

    return 0;
}
