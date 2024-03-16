# nanograd
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
