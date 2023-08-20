#include <cmath>
#include <iostream>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    std::cout << sigmoid(1) << std::endl;
    return 0;
}