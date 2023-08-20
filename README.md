# Neural Network in C++
A neural network written from scratch in C++

## NeuralNetwork

The NeuralNetwork class encapsulates the neural network's functionality.

### Private Members

    weights_hidden: A 2D vector representing the weights between the input layer and the hidden layer.
    weights_output: A vector representing the weights between the hidden layer and the output layer.
    bias_hidden: A bias term for the hidden layer.
    bias_output: A bias term for the output layer.
    gen: A Mersenne Twister random number generator.
    dis: A uniform distribution used to generate random numbers between -1 and 1.

### Constructor:

The constructor initializes the weights and biases with random values.

    Random Number Generation: The constructor initializes the Mersenne Twister random number generator (gen) and a uniform distribution (dis) that will generate random numbers between -1 and 1.

    Hidden Layer Weights: The weights_hidden 2D vector is resized to have hidden_nodes rows and input_nodes columns. Then, a nested loop is used to fill this matrix with random values between -1 and 1. These weights represent the connections between the input layer and the hidden layer.

    Output Layer Weights: The weights_output vector is resized to have output_nodes elements. A loop is used to fill this vector with random values between -1 and 1. These weights represent the connections between the hidden layer and the output layer.

    Biases: The bias_hidden and bias_output variables are initialized with random values between -1 and 1. Biases are used to shift the activation function to the left or right, helping the network make better approximations.Sigmoid Function

    The sigmoid function is an activation function used to introduce non-linearity into the network. It takes a value x and returns a value between 0 and 1.
### Forward Function:

    The forward function takes an input vector and performs a forward pass through the network, returning the output. It consists of the following steps:

    Hidden Layer Calculation: Multiplies the input by the hidden layer's weights, adds the bias, and applies the sigmoid activation function.
    Output Layer Calculation: Multiplies the hidden layer's output by the output layer's weights, adds the bias, and applies the sigmoid activation function.

###  The output:
The output of the network is a single value between 0 and 1, obtained by applying the sigmoid activation function. Since the network has not been trained, this value does not have a specific meaning or interpretation related to any real-world problem. It's simply the result of the mathematical transformations applied to the input.

### Other types of activation functions:
1. Parametric Sigmoid

You can introduce parameters to control the shape of the sigmoid function:

$$ \sigma(x) = \frac{1}{1+e^{a*x + b}} $$

Here, a controls the steepness of the curve, and b controls the horizontal shift.
2. Hyperbolic Tangent (tanh)

The tanh function is similar to the sigmoid but outputs values in the range (-1, 1):

$$ tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

3. Custom Activation Function

You can design your own custom activation function based on your specific needs. It could be a combination of existing functions or a completely new mathematical expression.

4. Leaky Sigmoid

A leaky version of the sigmoid can introduce a small slope for negative input values, which can sometimes help with training:

$$ \[
\sigma(x) =
\begin{cases}
\alpha \cdot x & \text{if } x < 0 \\
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0
\end{cases}
\]
$$

Here, `\(\alpha\)` is a small constant, such as 0.01.
### Example: Parametric Sigmoid
Here's how you might implement a parametric sigmoid in the existing code:
```cpp
double sigmoid(double x) {
    double a = 1.0; 
    double b = 0.0; 
    return 1.0 / (1.0 + exp(-(a * x + b)));
}
```
### Note
While it can be interesting to experiment with different activation functions, it's worth noting that the choice of activation function should be guided by the specific problem you're trying to solve and the properties of the data. More complex doesn't necessarily mean better, and well-established activation functions like ReLU, tanh, and the standard sigmoid are popular for good reasons. Always consider the problem context and empirical results when choosing or designing an activation function.

You can read more on activation functions [here](https://laid.delanover.com/activation-functions-in-deep-learning-sigmoid-relu-lrelu-prelu-rrelu-elu-softmax/)