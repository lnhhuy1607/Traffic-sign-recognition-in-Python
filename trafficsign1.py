import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
   
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Check for any images and labels
    if not images or not labels:
        sys.exit("No images or labels found. Check the data directory.")

    print(f"Number of images: {len(images)}")
    print(f"Number of labels: {len(labels)}")

    # Print a few labels to test
    print("Sample labels:", labels[:10])

    # Check that labels contain the appropriate number of categories
    unique_labels = set(labels)
    print(f"Unique labels: {unique_labels}")
    if len(unique_labels) > NUM_CATEGORIES:
        sys.exit(f"Number of unique labels ({len(unique_labels)}) exceeds NUM_CATEGORIES ({NUM_CATEGORIES})")
    
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    plot = model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file if provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
        plot_history(plot)

def plot_history(plot):
    # Get data about accuracy and error from plot
    accuracy = plot.history['accuracy']
    loss = plot.history['loss']
    
    # Get data about the accuracy and error of the validation set
    if 'val_accuracy' in plot.history:
        val_accuracy = plot.history['val_accuracy']
        val_loss = plot.history['val_loss']
    
    epochs = range(1, len(accuracy) + 1)
    
    # Plot an accuracy graph
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, '-', label='Training accuracy')
    if 'val_accuracy' in plot.history:
        plt.plot(epochs, val_accuracy, '-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot a loss graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, '-', label='Training loss')
    if 'val_loss' in plot.history:
        plt.plot(epochs, val_loss, '-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Path to data folder
    data_path = os.path.join(data_dir)

    # Loop through the subdirectories
    for label in range(NUM_CATEGORIES):
        sub_folder = os.path.join(data_path, str(label))

        for image_name in os.listdir(sub_folder):
            image_path = os.path.join(sub_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image / 255.0  # Normalize image to range [0, 1]
            images.append(image)
            labels.append(label)

    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([

        # Convolutional layers and Max-pooling layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Hidden Layers
        tf.keras.layers.Dense(128, activation='relu'),

        # Dropout
        tf.keras.layers.Dropout(0.5),

        # Output layer with output units for all digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    main()
