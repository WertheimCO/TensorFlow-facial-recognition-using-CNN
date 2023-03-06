# TensorFlow-facial-recognition-using-CNN
TensorFlow script for facial recognition using a convolutional neural network (CNN)

This script uses the CIFAR-10 dataset to train a CNN for facial recognition. The dataset contains 60,000 32x32 color images of 10 different classes of objects, including faces. The script preprocesses the images by normalizing the pixel values to be between 0 and 1. It then defines a CNN with three convolutional layers, two max pooling layers, and two dense layers. The last dense layer has 10 units, one for each class in the dataset. The model is compiled with the Adam optimizer and the sparse categorical cross-entropy loss function. Finally, the model is trained for 10 epochs on the training data and validated on the test data. The trained model is saved to a file called facial_recognition_model.h5.

To use this script, you will need to install TensorFlow and download the CIFAR-10 dataset.

pip install tensorflow

To download the CIFAR-10 dataset, you can use the following code:

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

Once you have installed TensorFlow and downloaded the dataset, you can run the script to train the model for facial recognition. Note that this script is a simple example and may not produce accurate results on real-world facial recognition tasks.



