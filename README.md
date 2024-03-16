# nanograd
Visualization of the computation graph, preserving the topological ordering used for the computation.

## SlowTorch 
* Neurons, Layers, MLP support, automatic instantiation with random intial weights
* Simple visualizations for the results and the loss are availiable

Low dimensional binary classification NN training run with visualization of predictions and loss found inside of `main_slowtorch.py`. 
![Image](/images/nn_val_and_loss.png)

## NanoTensor
* Autograd computation through Backpropagation
* PyTorch inspired functionality, support for multiple operations.
* Visualization of the Computation graph created by the backpropagation, preserving the topological sorting.

Computation graph and back propagation for a handful of chained operations in `main_nanotensor.py`
![Image](/images/computation_graph_total.png)

## Acknowledgements
Inspired by [micrograd](https://github.com/karpathy/micrograd).