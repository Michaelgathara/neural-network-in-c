# Trainable NN version

To run:
* Ensure you have C++ installed, Clang for example.
* Navigate to `test.cpp`
* Find the testing and training file and change these to csv's of your choice
* Find the following lines
```c
NeuralNetwork nn(5, 10, 1, 0.001);
NeuralNetwork nn(<input layer>, <hidden layer>, <output layer>, <Learning rate>);
nn.train(training_inputs, training_targets, 10);
```
* The first line, make sure the input layer is the size of the columns of your data, also make sure your data is numerical in nature
* You can customize the hidden, output layers and learning rate as you see fit
* Compile:
```sh
clang++ test.cpp trainable.cpp trainable.h
```
* Run the test.cpp
```sh
./a.out
```