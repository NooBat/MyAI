import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from NaiveImplementation import NaiveDense, NaiveSequential, fit

model = NaiveSequential([
    NaiveDense(input_size=784, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

assert(len(model.weights) == 4)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print('accuracy: %.2f' % np.average(matches))