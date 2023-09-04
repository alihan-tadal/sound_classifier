# Sound Classifier: CNN Model Trained on UrbanSound8k

This project contains a Convolutional Neural Network (CNN) model trained on the UrbanSound8k dataset. The model is designed for sound classification tasks and can be used to recognize different audio classes.

## About the Model

- **Model Type:** CNN (Convolutional Neural Network)
- **Library Used:** PyTorch
- **Training Dataset:** UrbanSound8k
- **Training Time:** Will be updated.
- **Model Performance:** Will be updated.

## Model Details

The implemented neural network, named `Network`, is designed for image classification tasks. Here are some key details about the model:

- **Architecture:** The model follows a standard Convolutional Neural Network (CNN) architecture consisting of four convolutional blocks, followed by flattening, a fully connected linear layer, and a softmax activation function for classification.

- **Convolutional Blocks:** There are four convolutional blocks in the network (`conv1`, `conv2`, `conv3`, and `conv4`). Each block includes a 2D convolutional layer, ReLU activation function, and max-pooling layer. These blocks help the model learn hierarchical features from input data.

- **Flatten Layer:** After the convolutional blocks, the output is flattened into a 1D tensor using the `nn.Flatten()` layer, preparing it for the fully connected layers.

- **Fully Connected Layer:** The flattened output is passed through a fully connected linear layer (`linear`) with 128 _ 5 _ 4 input features and 10 output features. This layer serves as the classifier for the different classes in your task.

- **Softmax Activation:** The model uses the softmax activation function to convert the raw logits into class probabilities. The `nn.Softmax(dim=1)` layer ensures that the probabilities sum to 1 along the class dimension.

- **Input Shape:** The model expects input data with a shape of (batch_size, 1, height, width), which is typical for grayscale images.

- **Output Shape:** The model produces class probabilities with a shape of (batch_size, num_classes), where `num_classes` is set to 10 in your current implementation.

Please note that this model is a basic example and may not be optimized for specific tasks. For improved performance, you may need to fine-tune hyperparameters, adjust the architecture, or use more complex models depending on your dataset and classification requirements.

## Training

Please use `train.py` file to train your model. After the training, model weights will be saved to the `cnn.pth` file. To learn how to use weights, please check `inference.py` file.

## Contact

Alihan Tadal
alihan.tadal@gmail.com
