# Caffe For FaceNet

## Modified Caffe Framework For FaceNet
What is **FaceNet**?

Link: [CVPR 2015 FaceNet: A Unified Embedding for Face Recognition and Clustering](http://arxiv.org/abs/1503.03832)

It is almost finished, but need further testing and bug fixing!!!

## Updates
#####15.7.26
- add **TripleLossLayer** class in loss_layers.hpp // empty class for now
- add **triplet_layer.cpp** and **triplet_layer.cu** //emtpy files, to be updated
#####15.8.13
- add **message TripleLossParameter** in **src/caffe/proto/caffe.proto**
#####15.8.14
- implemented **Forward_cpu**, **Backward_cpu**, **SetUpLayer** in **src/caffe/layers/triplet_layer.cpp**
#####15.8.15
- implemented **Forward_gpu**, **Backward_gpu** in **src/caffe/layers/triplet_layer.cu**
- and can be compiled successfully
#####15.8.16
- need further testing & fix bugs

## Caffe
Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like
