import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import tqdm

# slightly modified from Week 9 CNN lab 
input_size = 16
class CNNClassifier(nn.Module):
    def __init__(self, output_dim: int):
        super(CNNClassifier, self).__init__()
        assert output_dim > 0, "Output dimension must be a positive integer"
        self.conv1 = nn.Conv2d(
            in_channels = 200,
            out_channels = 16,
            kernel_size = (3, 1), 
            stride = (1, 1),
            padding = (1, 1)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (1,1)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 64, 
            kernel_size = (3, 3), 
            stride = (1, 1), 
            padding = (0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (1,1)
        )
        self.linear1 = nn.Linear(
            in_features=64,
            out_features=output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # reshape for linear layer
        # note that the output of maxpool 2 is (*,64,1,1) so we just need to take the first column and row. 
        # If the output size is not 1,1, we have to flatten x before going into linear using torch.flatten
        x = x[:,:,0,0] 
        x = self.linear1(x)     
        # x = torch.sigmoid(x)  -- removed the sigmoid function at the activation layer in order to turn this into a regression model 
        return x

# multimodal neural network model -- I attempted to make a multimodal neural network for Model #3 (ChatGPT helped me code this) 
class MultimodalNN:
    def __init__(self, image_dim, feature_dim, output_dim):
        # Define neural network layers for image data
        self.image_fc = self.init_fc_layer(image_dim, 128)

        # Define neural network layers for additional features
        self.feature_fc = self.init_fc_layer(feature_dim, 64)

        # Define combined feature dimension
        combined_dim = 128 + 64

        # Define final fully connected layers for prediction
        self.fc1 = self.init_fc_layer(combined_dim, 32)
        self.fc2 = self.init_fc_layer(32, output_dim)

    def init_fc_layer(self, input_dim, output_dim):
        # Initialize weights and biases for a fully connected layer
        weights = np.random.randn(input_dim, output_dim) * 0.01
        biases = np.zeros((1, output_dim))
        return {'weights': weights, 'biases': biases}

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def forward(self, image_input, feature_input):
        # Pass each input through its respective neural network layer
        image_output = self.relu(np.dot(image_input, self.image_fc['weights']) + self.image_fc['biases'])
        feature_output = self.relu(np.dot(feature_input, self.feature_fc['weights']) + self.feature_fc['biases'])

        # Concatenate the outputs
        combined_features = np.concatenate([image_output, feature_output], axis=1)

        # Pass through the final fully connected layers for prediction
        final_output = self.relu(np.dot(combined_features, self.fc1['weights']) + self.fc1['biases'])
        final_output = np.dot(final_output, self.fc2['weights']) + self.fc2['biases']

        return final_output



