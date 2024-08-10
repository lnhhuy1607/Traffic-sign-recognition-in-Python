import cv2
import csv
import numpy as np
import os
import sys
import tensorflow as tf

IMG_OUTPUT_WIDTH = 300
IMG_OUTPUT_HEIGHT = 300
TEXT_POSITION_H = 5
TEXT_POSITION_V = -10


def load_descriptions(csv_file):
    """Load csv of sign descriptions into a dictionary."""
    
    signs = {}

    with open(csv_file) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            # 1st column is the folder number, 2nd column is the description
            signs[int(row[0])] = row[1]

    return signs


def prep_image(image_path):
    """Read image and resize to match training image size."""
    
    img = cv2.imread(image_path)
    # Match training image size of 30x30
    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)

    return {'name': os.path.basename(image_path), 'array': img}


def add_prediction_to_image(image, prediction, sign_conversion):
    """Adds prediction and confidence level to the image dictionary."""

    # Take the highest result in the array, which is the class with the highest probability
    prediction_class = np.argmax(prediction)
    # The confidence is the probability of the image belonging to the predicted class
    confidence = "{:.0%}".format(prediction[prediction_class])

    image['prediction'] = prediction_class
    image['confidence'] = confidence
    image['complete'] = cv2.putText(
        cv2.resize(image['array'], (IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT)),
        sign_conversion[prediction_class].upper(),
        (TEXT_POSITION_H, IMG_OUTPUT_HEIGHT + TEXT_POSITION_V),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (180, 180, 0),
        thickness=2
    )

    return image


def main():

    # Check command-line arguments
    if len(sys.argv) < 3:
        sys.exit("Usage: python traffic_predict.py model image_path")

    # Load sign dictionary
    signs = load_descriptions('sign_descriptions.csv')

    # Prepare single image
    image_path = sys.argv[2]
    image = prep_image(image_path)

    # Load existing tf model from file
    model = tf.keras.models.load_model(sys.argv[1])

    # Predict image
    image_array = np.expand_dims(image['array'], axis=0)
    prediction = model.predict(image_array)[0]

    # Add prediction and confidence level to the image
    image = add_prediction_to_image(image, prediction, signs)

    # Display the image with prediction
    cv2.imshow('Image Prediction', image['complete'])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
