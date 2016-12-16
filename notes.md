# Notes from different tutorials
-----
These are notes that are collected from working through different tutorials and examples

## [Implementing a Neural Network from Scratch - An Introduction](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/#more-7)
* Number of nodes
    * The number of neurons in the input layer typically equals the number of features/dimensionality in the data (though sometimes there is a node for a bias term) [stackexchange](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
    * Number of nodes in the output layer is determined by the number of classes if using softmax
        * if regression then just one output node
        * if classification then either one output node or multiple depending on softmax or not
    * The number of nodes in the hidden layer will change the complexity of the functions we are able to fit. But it can lead to overfitting

