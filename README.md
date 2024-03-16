# nanograd
Nanograd is a custom backpropagation and neural net engine with syntax that is
similiar to PyTorch written for learning purposes.

The base object that takes the role of a Tensor is the NanoTenser which is a
wrapper around a single number. It supports computation and backpropagation
for most of the common mathematical operations (+, *, pow, exp, sin) and
activation functions (tanh, ReLu, sigmoid(Note yet)).

A simple MLP (Multiple Layer Perceptron) class named SlowTorch is implemented,
it allows easy initialization of arbitrary depth deep nets with random initial
weights and automatically assigns non-linearity activation functions to every
node that is not the final node.

Forward passes can be done simply by calling the training data as a function
input to the MLP (i.e. by executing model(x)), and backpropagation 
can be achieved by calling the `backward` function on any NanoTensor.

The gradients are then stored inside of the Tensor objects.

## SlowTorch 
* Neurons, Layers, MLP support, automatic instantiation with random intial weights
* Simple visualizations for the results and the loss are availiable

Binary Classification training results on the make_moons scipy training set
![Image](/images/make_moons_dataset.png)
![Image](/images/make_moons_results.png)

Low dimensional binary classification training results with visualization of predictions and loss found inside of `main_slowtorch.py`. 
![Image](/images/slowtorch_binary_classification.png)

## NanoTensor
* Autograd computation through Backpropagation
* PyTorch inspired functionality, support for multiple operations.
* Visualization of the Computation graph created by the backpropagation, preserving the topological sorting.

Computation graph and back propagation for a handful of chained operations in `main_nanotensor.py`
![Image](/images/computation_graph_total.png)

## Acknowledgements
Inspired by [micrograd](https://github.com/karpathy/micrograd).
