## Low Level Convolutional Neural Network with Parallelism

Author:
Brandon Xue brandonx@csu.fullerton.edu


### Introduction
This is a very basic convolutional neural network "from scratch", using lower level code such as the C programming language, and custom Cuda kernels.

Motivations:\
After discovering Keras in the previous fall term, I decided to deepen my understanding of deep learning by implementing a basic convolutional neural network from scratch. One thing I was dying to figure out was how to perform backpropagation without depending on some higher-level API. This project accomplishes that. Another reason for undertaking this project was to create an opportunity to practice Cuda programming, and to motivate the need for performance optimization. Last but not least, my professor, Dr. Bein, mentioned the massive speedup that could be achieved by using a GPU in deep learning tasks.

Disclaimer:\
I'm still learning about CNNs, and I have not taken any courses on deep learning. This project doesn't necessarily follow best practices; it is only a best-effort attempt.


### Current Results
- The sequential implementation (main branch) was compared with the parallel implementation (parallel branch) using the following neural network architecture:
  - input layer: receives 64 * 64 grayscale images that were normalized
    - Normalization was not per pixel, just per image
    - 64(width) x 64(height) x 1(channels) inputs, no batches
  - hidden layer 1: convolutional 2D
    - 16 filters/activation maps
    - 5 x 5 kernels, stride of 3 in each axis
    - ReLU activation
  - hidden layer 2: dense layer (based on my implementation, no need to flatten)
    - 128 output nodes
    - Sigmoid activation
  - output layer: dense layer
    - 15 output nodes (for the 15 classes of chinese characters)
    - This layer doesn't apply an activation, this is because I am using the softmax classifier (softmax activation + categorical cross entropy loss)
    - I can skip calculating the entire Jacobian matrix for the softmax classifier, and just calculate the partial derivative of loss with respect to this layer's outputs in one go if I handle the activation function separately.

- By a not-very-strict, naive benchmark, noted a speedup of over 18x compared to sequential implementation.
- The current weights uploaded to the repo are for a bigger network of 32 filters in hidden layer 1, and 512 output nodes in hidden layer 2.
  - This larger network yielded up to 98% accuracy.
  - (But this accuracy is from the model predicting on training data. No train/test split, yet.)


### Dependencies and shoutouts

- This project is written in C and Cuda, with some c++ dependencies.
  - C++ 14
  - gcc/g++ v7.5.0 
  - Cuda compilation tools V10.2.89
  - Test hardware: GeForce GTX TITAN X (Compute Capability 5.2)

- The data set used for this project is the Chinese MNIST numbers dataset. This is similar to the classic MNIST numbers, but had a slightly smaller and higher usability score.
  - https://www.kaggle.com/gpreda/chinese-mnist

- This project uses the wonderfully convenient stb_image.h single-header image library:
  - v2.26
  - https://github.com/nothings/stb

- Unit tests were a lifesaver. I decided to use Catch2, an elegant and handy unit-testing (mainly) framework.
  - v2.13.6
  - https://github.com/catchorg/Catch2


### Misc

- Here is a short slide show presented for the High Performance Computing class that I took:
- https://docs.google.com/presentation/d/1Vc6IWUTI6ZvKa9rwGm00YAWVQW1kl22JtiA7jIu-x9g/edit#slide=id.p


### Possible plans
- Work needs to be done on convolutional layers.
- Pooling layers need to be implemented.
- Trainable biases need to be implemented.
- Revisit Cuda kernels and take benchmarks. Optimize as needed.
- Add training optimizer, or change from stochastic gradient descent to batched or something.

More coming soon...
