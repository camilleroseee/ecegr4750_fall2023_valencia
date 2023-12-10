import torch
import math
import torch.nn as nn
from PIL import Image

# custom dataloader for images 
class ImageDataloader():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        x and y tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]
    def resize_images(self, desired_size):
        resized_images = []
        for img in self.x:
            resized_img = torch.nn.functional.interpolate(img.unsqueeze(0), size=desired_size, mode='bilinear', align_corners=False)
            resized_images.append(resized_img.squeeze(0))
        self.x = torch.stack(resized_images)
    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch':self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch':self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        # if the iter hasn't been generated yet
        if self.iter == None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)

        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        return batch


# dataloader for model 3 multimodal NN -- coded this with the help of ChatGPT
from sklearn.preprocessing import OneHotEncoder

class CustomMultimodalDataloader:
    def __init__(self, images, categorical_data, numerical_data, labels, batch_size=32, randomize=True):
        self.images = images
        self.categorical_data = categorical_data
        self.numerical_data = numerical_data
        self.labels = labels
        self.batch_size = batch_size
        self.randomize = randomize
        self.num_samples = len(images)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_indices = np.arange(self.num_samples)
        self.current_batch = 0

        # One-hot encode categorical features
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.categorical_data_encoded = self.encoder.fit_transform(categorical_data)

    def randomize_data(self):
        if self.randomize:
            np.random.shuffle(self.batch_indices)

    def generate_batch(self, start_idx):
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.batch_indices[start_idx:end_idx]

        batch_images = [self.images[idx] for idx in batch_indices]
        batch_categorical_data = self.categorical_data_encoded[batch_indices]
        batch_numerical_data = [self.numerical_data[idx] for idx in batch_indices]
        batch_labels = [self.labels[idx] for idx in batch_indices]

        return {
            'images': batch_images,
            'categorical_data': batch_categorical_data,
            'numerical_data': batch_numerical_data,
            'labels': batch_labels
        }

    def __iter__(self):
        self.current_batch = 0
        self.randomize_data()
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        batch_data = self.generate_batch(start_idx)
        self.current_batch += 1

        return batch_data





